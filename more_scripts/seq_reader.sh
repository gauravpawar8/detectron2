#!/bin/bash
perl -f /usr/local/lib/R/site-library/Thermimage/perl/split_fff.pl $1 
exiftool -RawThermalImage -b temp/*.fff > temp/thermalvid.raw
perl -f /usr/local/lib/R/site-library/Thermimage/perl/split_tiff.pl < temp/thermalvid.raw 
mv temp/*tiff $2
#mv temp/*fff $3 
rm -rf temp