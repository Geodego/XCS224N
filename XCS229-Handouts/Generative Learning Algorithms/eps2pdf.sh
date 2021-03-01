for filename in *eps; do
    echo "${filename%.*}"
    epstopdf $filename -o=${filename%.*}-eps-converted-to.pdf
done
