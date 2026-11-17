 def get_train_data_batch(
        self,
        completed_rollouts: list[CompletedRollout],
        *,
        global_steps: int = 0,
    ) -> tuple[DataProto, dict[str, Any]]:
        """Build a VERL training batch from completed rollouts."""
        level = self.trace_aggregator_level
        if level not in {"transition", "trajectory"}:
            raise ValueError(f"Unknown trace_aggregator_level: {level}")

        # Keep rollout randomness within each sample instead of ordering samples by completion time.
        sorted_rollouts = sorted(completed_rollouts, key=lambda rollout: (rollout.step, rollout.sample_idx_in_step))

        # [multimodal-patch] Image-to-row alignment is only implemented for the transition
        # level: trajectory-level aggregation merges multi-turn prompts into one row, which
        # breaks the correspondence between image placeholder tokens and image_urls, and
        # append_training_row is never given image_urls on that path. Fail loudly instead of
        # silently training without the vision signal.
        if level == "trajectory" and any(
            triplet.image_urls for rollout in sorted_rollouts for triplet in (rollout.triplets or [])
        ):
            raise ValueError(
                "[multimodal-patch] Rollout traces contain images (triplets with image_urls), but "
                "trace_aggregator level 'trajectory' merges multi-turn prompts and cannot keep the "
                "image-to-token alignment. Set agentlightning.trace_aggregator.level: transition "
                "in the config for multimodal training."
            )

        final_rewards: list[float] = []
        sample_with_reward_count = 0
        sample_with_trace_count = 0

        input_ids_list: list[list[int]] = []
        input_attention_mask_list: list[list[int]] = []
        response_ids_list: list[list[int]] = []
        response_attention_mask_list: list[list[int]] = []
        response_mask_list: list[list[int]] = []
        reward_list: list[float] = []
        data_id_list: list[str] = []
        rollout_id_list: list[str] = []
        turn_index_list: list[int] = []
        is_drop_list: list[bool] = []
        response_log_probs_list: list[list[float] | None] = []
        support_rows: list[dict[str, Any] | None] = []
        routed_experts_rows: list[tuple[str, int, int, int] | None] = []
        image_urls_list: list[list[str] | None] = []  # [multimodal-patch] per kept training row
        n_trunc_sample_because_of_response = 0
        n_skipped_empty_training_rows = 0
        unmerged_count = 0
        response_len_per_turn_list: list[int] = []
        merge_mismatch_rows: list[dict[str, Any]] = []

        def append_training_row(
            *,
            rollout_id: str,
            data_id: str,
            turn_index: int,
            prompt_ids: list[int],
            response_ids: list[int],
            reward: float,
            response_mask: list[int] | None = None,
            response_log_probs: list[float] | None = None,
            support: dict[str, Any] | None = None,
            routed_experts: str | None = None,
            image_urls: list[str] | None = None,  # [multimodal-patch]
        ) -> None:
            nonlocal n_skipped_empty_training_rows, n_trunc_sample_because_of_response
            original_prompt_length = len(prompt_ids)
            if len(prompt_ids) > self.max_prompt_length:
                prompt_ids = prompt_ids[: self.max_prompt_length]
                is_drop = True
            else:
                is_drop = False

            response_limit = self.max_response_length
            if self.max_total_length is not None:
                response_limit = min(response_limit, max(self.max_total_length - len(prompt_ids), 0))
            if len(response_ids) > response_limit:
                response_ids = response_ids[:response_limit]
                if response_mask is not None:
                    response_mask = response_mask[:response_limit]
                if response_log_probs is not None:
                    response_log_probs = response_log_probs[:response_limit]
                if support is not None:
                    support = {key: values[:response_limit] for key, values in support.items()}
                n_trunc_sample_because_of_response += 1

            if response_log_probs is not None and len(response_log_probs) != len(response_ids):
                response_log_probs = None

            train_token_count = sum(response_mask) if response_mask is not None else len(response_ids)
            if train_token_count == 0:
                n_skipped_empty_training_rows += 1
                return
            if self.require_routed_experts and routed_experts is None:
                raise RuntimeError(f"R3 requires routed_experts for rollout {rollout_id}")
            if self.require_score_centering:
                if support is None or response_log_probs is None:
                    raise ValueError(f"Missing score centering data for rollout {rollout_id}")
                if any(len(values) != len(response_ids) for values in support.values()):
                    raise ValueError("Score centering data and response token lengths differ")

            one_input_ids, one_input_attention_mask = get_left_padded_ids_and_attention_mask(
                prompt_ids, self.max_prompt_length, self.pad_token_id
            )
            one_response_ids, one_response_attention_mask = get_right_padded_ids_and_attention_mask(
                response_ids, self.max_response_length, self.pad_token_id
            )
            input_ids_list.append(one_input_ids)
            input_attention_mask_list.append(one_input_attention_mask)
            response_ids_list.append(one_response_ids)
            response_attention_mask_list.append(one_response_attention_mask)
            is_drop_list.append(is_drop)
            image_urls_list.append(image_urls)  # [multimodal-patch] stays aligned with kept rows
            if response_mask is not None:
                one_response_mask, _ = get_right_padded_ids_and_attention_mask(
                    response_mask, self.max_response_length, 0
                )
                response_mask_list.append(one_response_mask)

            response_log_probs_list.append(response_log_probs)
            support_rows.append(support)
            routed_experts_rows.append(
                (routed_experts, original_prompt_length, len(prompt_ids), len(response_ids))
                if routed_experts is not None
                else None
            )

            reward_list.append(reward)
            data_id_list.append(data_id)
            rollout_id_list.append(rollout_id)
            if level == "transition":
                turn_index_list.append(turn_index)

        for rollout in sorted_rollouts:
            final_reward = self._fillna_reward(rollout)
            final_rewards.append(final_reward)
            if rollout.final_reward is not None:
                sample_with_reward_count += 1

            if not rollout.triplets:
                print(f"Warning: No triplets found for training rollout {rollout.rollout_id}, skipping.")
                continue
            sample_with_trace_count += 1

            if level == "transition":
                for turn_index, triplet in enumerate(rollout.triplets):
                    response_ids = triplet.response["token_ids"]
                    log_probs = triplet.response["log_probs"]
                    response_len_per_turn_list.append(len(response_ids))
                    append_training_row(
                        rollout_id=rollout.rollout_id,
                        data_id=rollout.data_id,
                        turn_index=turn_index,
                        prompt_ids=triplet.prompt["token_ids"],
                        response_ids=response_ids,
                        reward=final_reward,
                        response_log_probs=log_probs,
                        support=_score_centering(triplet.response),
                        routed_experts=triplet.response.get("routed_experts"),
                        image_urls=triplet.image_urls,  # [multimodal-patch]
                    )
                continue
            else:
                first_triplet = rollout.triplets[0]
                group_start_turn_index = 0
                current_prompt_ids = list(first_triplet.prompt["token_ids"])
                current_response_ids = list(first_triplet.response["token_ids"])
                current_context = current_prompt_ids + current_response_ids
                current_response_mask = [1] * len(current_response_ids)
                current_response_log_probs: list[float] | None = first_triplet.response["log_probs"]
                if current_response_log_probs is not None:
                    current_response_log_probs = list(current_response_log_probs)
                current_routed_experts = first_triplet.response.get("routed_experts")
                current_support = _score_centering(first_triplet.response)
                response_len_per_turn_list.append(len(current_response_ids))
                merged_group_count = 0

                for turn_index, triplet in enumerate(rollout.triplets[1:], start=1):
                    prompt_ids = triplet.prompt["token_ids"]
                    response_ids = triplet.response["token_ids"]
                    log_probs = triplet.response["log_probs"]
                    response_len_per_turn_list.append(len(response_ids))
                    next_context = prompt_ids + response_ids

                    if ids_startswith(prompt_ids, current_context):
                        if len(prompt_ids) > len(current_context):
                            observation_ids = prompt_ids[len(current_context) :]
                            current_response_ids += observation_ids
                            current_response_mask += [0] * len(observation_ids)
                            if current_response_log_probs is not None:
                                current_response_log_probs += [0.0] * len(observation_ids)
                            if current_support is not None:
                                for values in current_support.values():
                                    values.extend([[] for _ in observation_ids])
                        current_response_ids += response_ids
                        current_response_mask += [1] * len(response_ids)
                        if current_response_log_probs is not None:
                            if log_probs is None or len(log_probs) != len(response_ids):
                                current_response_log_probs = None
                            else:
                                current_response_log_probs += list(log_probs)
                        current_context = next_context
                        next_support = _score_centering(triplet.response)
                        if current_support is not None and next_support is not None:
                            for key in current_support:
                                current_support[key].extend(next_support[key])
                        else:
                            current_support = None
                        current_routed_experts = triplet.response.get("routed_experts")
                        continue

                    if len(merge_mismatch_rows) < _TRACE_MERGE_MISMATCH_WANDB_LIMIT:
                        mismatch_row = {
                            "global_steps": global_steps,
                            "rollout_id": rollout.rollout_id,
                            "data_id": rollout.data_id,
                            "turn_index": turn_index,
                            "prompt_length": len(prompt_ids),
                            "response_length": len(response_ids),
                            "previous_trace_length": len(current_context),
                            "current_trace_length": len(next_context),
                        }
                        if self.classify_trace_merge_mismatches:
                            template_mismatch, retoken_mismatch, others_mismatch = _classify_trace_merge_mismatch(
                                self.tokenizer, current_context, prompt_ids
                            )
                            mismatch_row.update(
                                template_mismatch=template_mismatch,
                                retoken_mismatch=retoken_mismatch,
                                others_mismatch=others_mismatch,
                            )
                        mismatch_row.update(
                            previous_trace=_decode_trace_text(self.tokenizer, current_context),
                            current_trace=_decode_trace_text(self.tokenizer, next_context),
                        )
                        merge_mismatch_rows.append(mismatch_row)

                    append_training_row(
                        rollout_id=rollout.rollout_id,
                        data_id=rollout.data_id,
                        turn_index=group_start_turn_index,
                        prompt_ids=current_prompt_ids,
                        response_ids=current_response_ids,
                        reward=final_reward,
                        response_mask=current_response_mask,
                        response_log_probs=current_response_log_probs,
                        support=current_support,
                        routed_experts=current_routed_experts,
                    )
                    merged_group_count += 1

                    group_start_turn_index = turn_index
                    current_context = next_context
                    current_prompt_ids = list(prompt_ids)
                    current_response_ids = list(response_ids)
                    current_response_mask = [1] * len(response_ids)
                    current_response_log_probs = list(log_probs) if log_probs is not None else None
                    current_routed_experts = triplet.response.get("routed_experts")
                    current_support = _score_centering(triplet.response)

                append_training_row(
                    rollout_id=rollout.rollout_id,
                    data_id=rollout.data_id,
                    turn_index=group_start_turn_index,
                    prompt_ids=current_prompt_ids,
                    response_ids=current_response_ids,
                    reward=final_reward,
                    response_mask=current_response_mask,
                    response_log_probs=current_response_log_probs,
                    support=current_support,
                    routed_experts=current_routed_experts,
                )
                merged_group_count += 1

                if merged_group_count > 1:
                    unmerged_count += 1

        rollout_trajectory_records = _build_compact_rollout_trajectory_records(
            sorted_rollouts,
            tokenizer=self.tokenizer,
            reward_fillna_value=self.reward_fillna_value,
            limit=_ROLLOUT_TRAJECTORY_WANDB_LIMIT,
        )
        _upload_trace_merge_mismatches_to_wandb(
            merge_mismatch_rows, global_steps, include_classification=self.classify_trace_merge_mismatches
        )
        _upload_compact_rollout_trajectories_to_wandb(rollout_trajectory_records, global_steps)

        n_sample = len(input_ids_list)
        if n_sample == 0:
            raise RuntimeError("get_train_data_batch emitted zero training rows.")

        batch_input_ids = torch.LongTensor(input_ids_list).to(self.device)
        input_attention_mask = torch.LongTensor(input_attention_mask_list).to(self.device)
        batch_response_ids = torch.LongTensor(response_ids_list).to(self.device)
        response_attention_mask = torch.LongTensor(response_attention_mask_list).to(self.device)
        batch_response_mask = torch.LongTensor(response_mask_list).to(self.device) if level == "trajectory" else None

        batch_seq = torch.cat([batch_input_ids, batch_response_ids], dim=-1)
        attention_mask = torch.cat([input_attention_mask, response_attention_mask], dim=-1)
        position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)

        # [multimodal-patch] Build per-row multi_modal_inputs and mrope position ids.
        has_image_rows = any(image_urls for image_urls in image_urls_list)
        multi_modal_inputs_list: list[dict[str, Any] | None] | None = None
        if has_image_rows and self.processor is None:
            for row_index, image_urls in enumerate(image_urls_list):
                if image_urls:
                    is_drop_list[row_index] = True
            print("Warning: [multimodal-patch] marking image rows for drop because RolloutAdapter has no processor.")
        elif has_image_rows:
            use_mrope = _is_mrope_processor(self.processor)
            if not use_mrope:
                print(
                    "Warning: [multimodal-patch] processor is not a recognized mrope (Qwen-VL) "
                    "processor; multi_modal_inputs will be attached but position_ids stay 2D."
                )
            multi_modal_inputs_list = []
            mrope_position_ids_list: list[torch.Tensor] = []
            for row_index in range(n_sample):
                image_urls = image_urls_list[row_index]
                row_multi_modal_inputs: dict[str, Any] | None = None
                row_image_grid_thw = None
                row_has_mm_token_type_ids = False
                if image_urls and is_drop_list[row_index]:
                    # The over-long prompt was truncated and may have cut through the image
                    # placeholder tokens, so the image grid metadata no longer matches the
                    # token sequence. Fall back to a text-only row; this is safe because
                    # is_drop rows are filtered out by is_drop_mask before the training
                    # forward (see trainer.py) and never contribute gradients.
                    print(
                        f"Warning: [multimodal-patch] row {row_index} (rollout "
                        f"{rollout_id_list[row_index]}) has a truncated (is_drop) prompt with "
                        "images; falling back to text-only for this row."
                    )
                    image_urls = None
                if image_urls:
                    try:
                        row_multi_modal_inputs, row_has_mm_token_type_ids = _build_multi_modal_inputs(
                            self.processor, image_urls
                        )
                        row_image_grid_thw = row_multi_modal_inputs.get("image_grid_thw")
                    except Exception as exc:
                        is_drop_list[row_index] = True
                        print(
                            f"Warning: [multimodal-patch] image processing failed; marking row {row_index} for drop "
                            f"(rollout {rollout_id_list[row_index]}): {exc}"
                        )
                        row_multi_modal_inputs = None
                multi_modal_inputs_list.append(row_multi_modal_inputs)
                if use_mrope:
                    if image_urls and row_image_grid_thw is None and row_multi_modal_inputs is not None:
                        is_drop_list[row_index] = True
                        print(
                            f"Warning: [multimodal-patch] row {row_index} (rollout "
                            f"{rollout_id_list[row_index]}) is missing image_grid_thw; marking it for drop."
                        )
                    try:
                        mrope_position_ids_list.append(
                            _compute_mrope_position_ids(
                                self.processor,
                                input_ids=batch_seq[row_index],
                                attention_mask=attention_mask[row_index],
                                image_grid_thw=row_image_grid_thw,
                                has_mm_token_type_ids=row_has_mm_token_type_ids,
                            )
                        )
                    except Exception as exc:
                        if image_urls:
                            is_drop_list[row_index] = True
                        print(
                            f"Warning: [multimodal-patch] mrope position ids failed for row {row_index} "
                            f"(rollout {rollout_id_list[row_index]}), using text-only variant: {exc}"
                        )
                        try:
                            mrope_position_ids_list.append(
                                _compute_mrope_position_ids(
                                    self.processor,
                                    input_ids=batch_seq[row_index],
                                    attention_mask=attention_mask[row_index],
                                    image_grid_thw=None,
                                )
                            )
                        except Exception:
                            # Last resort (e.g. leftover image tokens in a truncated prompt make
                            # even the text variant of get_rope_index fail): plain cumsum positions
                            # on all four mrope rows. The row is almost always an is_drop row that
                            # is filtered before the training forward anyway.
                            print(
                                f"Warning: [multimodal-patch] text-only mrope fallback also failed for "
                                f"row {row_index} (rollout {rollout_id_list[row_index]}); "
                                "using plain cumsum position ids."
                            )
                            mrope_position_ids_list.append(
                                _text_only_mrope_position_ids(batch_seq[row_index], attention_mask[row_index])
                            )
            if use_mrope:
                # (n_sample, 4, seq_len): verl's engine detects mrope via position_ids.dim() == 3.
                position_ids = torch.stack(mrope_position_ids_list, dim=0)

        row_has_log_probs_list = [log_probs is not None for log_probs in response_log_probs_list]
        emit_rollout_log_probs = all(row_has_log_probs_list)
        if not emit_rollout_log_probs and any(row_has_log_probs_list):
            print("Warning: Mixed rollout log_probs availability, omitting rollout_log_probs from batch.")

        is_drop_mask = torch.BoolTensor(is_drop_list).to(self.device)
        # Preserve fractional reward differences used by advantage estimators.
        scores = torch.tensor(reward_list, dtype=torch.float32).to(self.device)

        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)
        token_positions = torch.arange(attention_mask.shape[-1], device=attention_mask.device).unsqueeze(0)
        eos_mask_idx = torch.argmax(token_positions * attention_mask, dim=-1)
        token_level_scores[torch.arange(n_sample), eos_mask_idx] = scores
        token_level_scores = token_level_scores[:, -self.max_response_length :]

        batch_dict = {
            "prompts": batch_input_ids,
            "responses": batch_response_ids,
            "input_ids": batch_seq,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "is_drop_mask": is_drop_mask,
            "token_level_scores": token_level_scores.contiguous(),
        }
        if level == "trajectory":
            assert batch_response_mask is not None
            batch_dict["response_mask"] = batch_response_mask
        if emit_rollout_log_probs:
            padded_log_probs_list = [
                log_probs + [0.0] * (self.max_response_length - len(log_probs))
                for log_probs in response_log_probs_list
                if log_probs is not None
            ]
            batch_dict["rollout_log_probs"] = torch.tensor(padded_log_probs_list, dtype=torch.float32).to(self.device)
        if routed_experts_rows and all(row is not None for row in routed_experts_rows):
            batch_dict["routed_experts"] = _build_routed_experts_batch(
                [row for row in routed_experts_rows if row is not None],
                self.max_prompt_length,
                self.max_response_length,
                self.device,
            )

        batch = TensorDict(batch_dict, batch_size=n_sample)  # type: ignore[arg-type]
        data_proto = DataProto(batch=batch)
        data_proto.non_tensor_batch["data_id_list"] = np.array(data_id_list)
        data_proto.non_tensor_batch["rollout_id_list"] = np.array(rollout_id_list)
        if self.require_score_centering:
            # Preserve variable supports through row selection, rank dispatch and
            # microbatch slicing; pad only the active positions inside the worker.
            data_proto.non_tensor_batch["score_centering"] = np.array(support_rows, dtype=object)
        if level == "transition":
            data_proto.non_tensor_batch["turn_index_list"] = np.array(turn_index_list)
        if multi_modal_inputs_list is not None:
            # [multimodal-patch] Per-row dict (or None for text rows), matching
            # verl 0.8.0 extract_multi_modal_inputs expectations.
            data_proto.non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        n_response_turns = len(response_len_per_turn_list)
        data_metrics = {
            "training/reward": float(np.mean(final_rewards)) if final_rewards else 0.0,
            "training/n_sample": n_sample,
            "training/n_rollouts": len(sorted_rollouts),
            "training/n_rollouts_w_trace": sample_with_trace_count,
            "training/n_rollouts_w_reward": sample_with_reward_count,
            "training/n_truncated_sample": n_trunc_sample_because_of_response,
            "training/n_skipped_empty_rows": n_skipped_empty_training_rows,
            "training/n_turns": n_response_turns,
            "response_length/training/avg_by_turn": float(np.mean(response_len_per_turn_list)),
            "response_length/training/max_by_turn": int(np.max(response_len_per_turn_list)),
            "response_length/training/min_by_turn": int(np.min(response_len_per_turn_list)),
        }
        if level == "trajectory":
            data_metrics["training/n_unmerged_rollouts"] = unmerged_count
            data_metrics["training/n_trace_merge_mismatch_rows"] = len(merge_mismatch_rows)

        return data_proto, data_metrics
