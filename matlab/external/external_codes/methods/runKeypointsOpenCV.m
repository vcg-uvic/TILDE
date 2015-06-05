function [] = runKeypointsOpenCV(method_name, in_file_name, out_file_name)
global sRoot;
       
    [status, whereisOpenCV] = system('/usr/local/bin/pkg-config opencv --libs');
    library_path_linux = sprintf([ 'LD_LIBRARY_PATH="' whereisOpenCV '" ; export LD_LIBRARY_PATH']);
    library_path_mac = sprintf([ 'DYLD_LIBRARY_PATH="' whereisOpenCV '" ; export DYLD_LIBRARY_PATH']);
    library_path = [library_path_mac ';' library_path_linux ';'];
    
       if (status ~= 0)
            library_path = 'LD_LIBRARY_PATH="/cvlabdata1/cvlab/OpenCV-2.4.6.1/install/lib/" ; export LD_LIBRARY_PATH;';
            warning('using server hard coded path')
%            error('whitout knowing where is opencv lib, I cannot do anything..., do you have pkg-config ?')
       end

%   ?LD_PRELOAD=/cvlabdata1/cvlab/OpenCV-2.4.6.1/install/lib/libopencv_core.so.2.4:/cvlabdata1/cvlab/OpenCV-2.4.6.1/install/lib/libopencv_imgproc.so.2.4:/cvlabdata1/cvlab/OpenCV-2.4.6.1/install/lib/libopencv_features2d.so.2.4:/cvlabdata1/cvlab/OpenCV-2.4.6.1/install/lib/libopencv_nonfree.so.2.4 
    detector_path = [sRoot '/external/external_codes/methods/opencvKeypointDetector'];
    com = sprintf('%s %s %s %s %s',library_path, detector_path, method_name, in_file_name, out_file_name );

    system(com);

end