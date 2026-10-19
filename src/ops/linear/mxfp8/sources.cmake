target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/mxfp8_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mxfp8_gemv.cu"
  "${CMAKE_CURRENT_LIST_DIR}/mxfp8_mma.cu"
)
