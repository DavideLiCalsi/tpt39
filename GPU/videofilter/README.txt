There are two files, both working videofilters, but filter.cpp does not use mapped buffers, while filter_opt.cpp uses mapped buffers.
In order to test one of them, you just need to change the source field in the makefile. Currently it is set to the version with mapped buffers (filter_opt.cpp).
