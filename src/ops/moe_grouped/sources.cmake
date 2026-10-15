target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/moe_grouped.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kernels.cu"
)
