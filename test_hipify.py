from hipify_torch import hipify_python
import os

print ("HIPIFICATION IS IN PROGRESS....")
#this_dir = os.path.dirname(os.path.abspath(__file__))
source_root = os.path.abspath(os.path.dirname(__file__))
proj_dir = os.path.join(source_root, "cupy_backends/cuda/")
api_path = os.path.join(source_root, "cupy_backends/cuda/api")
libs_path = os.path.join(source_root, "cupy_backends/cuda/libs")
output_dir = os.path.join(source_root, "cupy_backends", "hip")
## hipify cupy_backends/cuda

ignores_hipify_1 = [
                    api_path + "/*",
                    libs_path + "/*",
                   ]

print (proj_dir)

with hipify_python.GeneratedFileCleaner(keep_intermediates=True) as clean_ctx:
    hipify_python.hipify(
        project_directory=proj_dir,
        output_directory=output_dir,
        includes=['*'],
        show_detailed=True,
        #ignores=ignores_hipify_1,
        header_include_dirs=[],
        is_pytorch_extension=True,
        clean_ctx=clean_ctx,
        )
