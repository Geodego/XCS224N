mkdir -p bak
for filename in bak/*original-version; do
    #echo "${filename%.*}"
    echo "processing $filename"
    temp=${filename%.original-version}
    echo "new file name ${temp#bak/}"
    cp $filename ${temp#bak/}
done
