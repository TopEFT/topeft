for f in 2022_*.json; do
    base="${f#2022_}"       # remove '2022_'
    mv "$f" "${base%.json}_2022.json"
done
