RENDERER = dict(
    DIFF_RENDERER="DIBR",
    RENDER_TYPE="batch",  # batch | batch_tex | batch_single | batch_single_tex
    DIBR=dict(
        ZNEAR=0.01,
        ZFAR=100.0,
        HEIGHT=480,
        WIDTH=640,
        MODE="VertexColorBatch",  # VertexColorMulti | VertexColorBatch | TextureBatch | TextureMulti
    ),
)
