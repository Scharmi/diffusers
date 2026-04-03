# def ays():
#     model = PredictorUNet(
#         T=PREDICTOR_T,
#         suffix=f"_{DATASET_CONFIG_NAME}",
#         n_channels=dataset_config["channels"],  # ty: ignore
#         img_width=dataset_config["img_width"],  # ty: ignore
#         img_height=dataset_config["img_height"],  # ty: ignore
#     ).cuda()
#     model.eval()

#     logger.info(f"Using schedule config: {SCHEDULE_CONFIG_NAME}")
#     logger.info(f"Using denoiser config: {SOLVER_CONFIG_NAME}")
#     logger.info(f"Using equation config: {EQUATION_CONFIG_NAME}")

#     equation = equation_config(
#         model=model,
#         schedules=schedules,
#     )

#     solver: Solver = solver_config(
#         equation=equation,
#         T=SOLVER_T,
#     )

#     ays_schedule = AYSSamplingSchedule(
#         denoiser=solver,
#         dataloader=get_dataloader(
#             batch_size=BATCH_SIZE,
#             dataset_class=dataset_config["class"],  # ty: ignore
#             width=dataset_config["img_width"],  # ty: ignore
#             height=dataset_config["img_height"],  # ty: ignore
#         ),
#         config=AYSConfig(
#             max_iter=1,
#             max_finetune_iter=1,
#             save_file=f"generated/ays_timesteps_{SOLVER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}.pt",
#         ),
#     )

#     try:
#         initial_t = torch.load(
#             f"generated/ays_timesteps_{SOLVER_CONFIG_NAME}_{SCHEDULE_CONFIG_NAME}.pt_10",
#             weights_only=False,
#         )
#     except FileNotFoundError:
#         initial_t = None

#     logger.info(f"Starting AYS tuning with initial_t: {initial_t}")
#     timesteps = ays_schedule.get_20_timesteps(initial_t)  # ty: ignore
#     logger.info(f"Final timesteps: {timesteps.steps}")
