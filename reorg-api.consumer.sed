#
# Pass this script to Sed via a command such as:
#    find . \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' \) -print | xargs sed -Ef reorg-api.consumer.sed -i
#

s/#include "(tt_metal\/)?common\/(assert.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?common\/(base_types.hpp)"/#include <tt-metalium\/\2>/
s/#include <(tt_metal\/)?common\/(base_types.hpp)>/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?common\/(constants.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?common\/(core_coord.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?common\/(logger.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?common\/(tt_backend_api_types.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?(host_api.hpp)"/#include <tt-metalium\/\2>/
s/#include <(tt_metal\/)?(host_api.hpp)>/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/allocator\/(allocator.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?tt_stl\/(reflection.hpp)"/#include <tt-metalium\/\2>/



s/#include "(tt_metal\/)?common\/(bfloat16.hpp)"/#include <tt-metalium\/\2>/
s/#include "(common\/)(bfloat16.hpp)"/#include <tt-metalium\/\2>/
s/#include <(common\/)(bfloat16.hpp)>/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?common\/(bfloat4.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?common\/(bfloat8.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?common\/(test_tiles.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?detail\/(tt_metal.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?graph\/(graph_tracking.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/buffers\/(buffer.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/buffers\/(global_circular_buffer.hpp)"/#include <tt-metalium\/global_circular_buffer_impl.hpp>/
s/#include "(tt_metal\/)?impl\/device\/(device.hpp)"/#include <tt-metalium\/device_impl.hpp>/
s/#include "(tt_metal\/impl\/)?device\/(device.hpp)"/#include <tt-metalium\/device_impl.hpp>/
s/#include <(tt_metal\/)?impl\/device\/(device.hpp)>/#include <tt-metalium\/device_impl.hpp>/
s/#include "(tt_metal\/)(device.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?tt_stl\/(concepts.hpp)"/#include <tt-metalium\/device_impl.hpp>/
s/#include "(tt_metal\/)?include\/tt_metal\/(global_circular_buffer.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/buffers\/(global_semaphore.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?tt_stl\/(span.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/tt_stl\/)?(span.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/sub_device\/(sub_device.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?distributed\/(mesh_device.hpp)"/#include <tt-metalium\/\2>/
s/#include <(tt_metal\/)?distributed\/(mesh_device_view.hpp)>/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?distributed\/(mesh_device_view.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/tile\/(tile.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?common\/(math.hpp)"/#include <tt-metalium\/\2>/
s/#include <(tt_metal\/)?common\/(math.hpp)>/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/dispatch\/(command_queue.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/buffers\/(buffer_constants.hpp)"/#include <tt-metalium\/\2>/
s/#include "(buffers\/)(buffer_constants.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/program\/(program.hpp)"/#include <tt-metalium\/program_impl.hpp>/
s/#include "(tt_metal\/)?tt_stl\/(type_name.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/device\/(device_pool.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/device\/(program_cache.hpp)"/#include <tt-metalium\/\2>/
# s/#include "(tt_metal\/)?tools\/profiler\/(op_profiler.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?tools\/profiler\/(profiler.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?common\/(work_split.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?detail\/(util.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/buffers\/(circular_buffer.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/trace\/(trace.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/buffers\/(circular_buffer_types.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/kernels\/(kernel_types.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/impl\/)?kernels\/(kernel_types.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?detail\/reports\/(compilation_reporter.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?detail\/reports\/(memory_reporter.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/kernels\/(kernel.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?llrt\/(rtoptions.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?common\/(tilize_untilize.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?(tt_log.h)"/#include <tt-metalium\/\2>/
s/#include "sub_device\/(sub_device_types.hpp)"/#include <tt-metalium\/\1>/
s/#include "allocator\/(allocator.hpp)"/#include <tt-metalium\/\1>/
s/#include "(tt_metal\/)?impl\/kernels\/(runtime_args_data.hpp)"/#include <tt-metalium\/\2>/
s/#include "tt_metal\/third_party\/tracy\/public\/(tracy\/Tracy.hpp)"/#include <\1>/
s/#include "(tt_metal\/)?experimental\/(hal.hpp)"/#include <tt-metalium\/hal_exp.hpp>/
s/#include "(tt_metal\/)?tt_stl\/(overloaded.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?impl\/event\/(event.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?detail\/(persistent_kernel_cache.hpp)"/#include <tt-metalium\/\2>/
s/#include "(tt_metal\/)?llrt\/(tt_cluster.hpp)"/#include <tt-metalium\/\2>/
s/#include "tt_metal\/hostdevcommon\/api\/hostdevcommon\/(common_values.hpp)"/#include <hostdevcommon\/\1>/
s/#include "tt_metal\/hostdevcommon\/(common_values.hpp)"/#include <hostdevcommon\/\1>/
s/#include "tt_metal\/hw\/inc\/(risc_attribs.h)"/#include <tt-metalium\/\1>/
s/#include "tt_metal\/hw\/inc\/(dataflow_api.h)"/#include <tt-metalium\/\1>/
s/#include "tt_metal\/distributed\/(system_mesh.hpp)"/#include <tt-metalium\/\1>/
