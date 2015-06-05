This package contains matlab files for computing repeatability and matching score between two images [ref. performance evaluation papers].
http://www.robots.ox.ac.uk/~vgg/research/affine
km@robots.ox.ac.uk

Recompliling c_eoverlap.cxx  and descdist.cxx might be necessary
matlab>>mex c_eoverlap.cxx;
matlab>>mex descdist.cxx;


To compute the repeatability score [ref. detector evaluation paper]:

[v_overlap,v_repeatability,v_nb_of_corespondences,matching_score,nb_of_matches]=repeatability('img1.haraff.sift','img2.haraff.sift','H1to2p','img1.ppm','img2.ppm',1);

v_overlap - overlap errors used for estimation of the repeatability.

v_repeatability - repeatability estimated for given overlap errors

v_nb_of_corespondences - number of corresponding regions for given overlap errors

matching_score - matching score for a given overlap error

nb_of_matches -  number of correct matches for a given overlap error

To evaluate desciriptor performance on a pair of images [ref. descriptor evaluation paper]:

run repeatability to compute the correspondence matrix
[v_overlap,v_repeatability,v_nb_of_corespondences,matching_score,v_nb_of_matches,twi]=repeatability('img1.haraff.sift','img2.haraff.sift','H1to2p','img1.ppm','img2.ppm',0);

then run descperf to compute the matching score
[correct_match_nn, total_match_nn,correct_match_sim,total_match_sim,correct_match_rn,total_match_rn]=descperf('img1.haraff.sift','img2.haraff.sift','H1to2p','img1.ppm','img2.ppm',v_nb_of_corespondences(5),twi);


correct_match_nn - number of correct matches with nearest neighbour matching strategy
total_match_nn - total number of matches with nearest neighbour matching strategy

correct_match_sim - number of correct matches with threshold based matching strategy
total_match_sim - total number of matches with  threshold based matching strategy


correct_match_rn - number of correct matches with  nearest neighbour distance ratio matching strategy
total_match_rn - total number of matches with  nearest neighbour distance ratio matching strategy

