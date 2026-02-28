# DA3-Large inference config (mmlab style)
da3_large_cfg = dict(
    model=dict(
        variant='da3-large',
        backbone=dict(
            name='vitl',
            out_layers=[11, 15, 19, 23],
            alt_start=8,
            qknorm_start=8,
            rope_start=8,
            cat_token=True,
        ),
        head=dict(
            dim_in=2048,
            output_dim=2,
            features=256,
            out_channels=[256, 512, 1024, 1024],
        ),
        cam_enc=dict(dim_out=1024),
        cam_dec=dict(dim_in=2048),
    ),
    input_processor=dict(
        process_res=504,
        process_res_method='upper_bound_resize',
        patch_size=14,
    ),
    output_processor=dict(),
    weights='/home/drobotics/bohaozhang/da3_large_weights.pth',
    device='cuda',
)
