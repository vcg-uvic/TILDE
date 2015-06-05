function [failed] = sifer(in, out)
failed = false;

global sRoot;

	exportvar = sprintf(['DETECTMETHODS="' sRoot '/external/external_codes/methods" ; export DETECTMETHODS']);
	pathvar = sprintf(['PATH="$PATH:/usr/local/bin:' sRoot '/external/external_codes/methods" ; export PATH']);

    rootFolder = [ sRoot '/external/external_codes/methods/'];
    full_path_exec = [rootFolder 'sifer.sh'];
    
    if (exist(full_path_exec) ~= 2)
        failed = true;
        return;
    end
    
    
    com = sprintf([pathvar ';' full_path_exec ' ' in ' ' rootFolder ' > ' out] );
    status = system(com);

    if status ~= 0
        warning ('command did not worked, are you running Sifer compiled on the right architecture ?')
    end
    
	% second run to get sifer raw output
    com = sprintf([pathvar ';' rootFolder 'sifer_raw.sh ' in ' ' rootFolder ' > ' out '.raw'] );
    status = system(com);
    
    if status ~= 0
        warning ('command did not worked, are you running Sifer compiled on the right architecture ?')
    end
    

    % load the raw output
    sifer_raw = load([out '.raw']);
    sifer_score = abs(sifer_raw(:,end));

    % save the scores
	score_file = fopen([out '.score'], 'w');
	for ii = 1:length(sifer_score)
	    fprintf(score_file, '%f\n', sifer_score(ii));
	end
	fclose(score_file);

end