target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/moe_slots_gguf.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/decode_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/grouped_kernels.cu"
)
