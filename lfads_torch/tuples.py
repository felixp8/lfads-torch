from collections import namedtuple

SessionBatch = namedtuple(
    "SessionBatch",
    [
        "encod_data",
        "recon_data",
        "encod_mask",
        "recon_mask",
        "temp_context",
        "ext_input",
        "truth",
        "sv_mask",
        "encod_mean_std",
        "recon_mean_std",
    ],
)

SessionOutput = namedtuple(
    "SessionOutput",
    [
        "output_params",
        "factors",
        "ic_mean",
        "ic_std",
        "co_means",
        "co_stds",
        "gen_states",
        "gen_init",
        "gen_inputs",
        "con_states",
    ],
)
