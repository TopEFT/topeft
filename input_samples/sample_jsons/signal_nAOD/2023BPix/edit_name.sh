for f in 2023BPix_*.json; do
    base="${f#2023BPix_}"       # remove '2023BPix_'
    mv "$f" "${base%.json}_2023BPix.json"
done
