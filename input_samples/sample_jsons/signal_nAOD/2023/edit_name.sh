for f in 2023_*.json; do
    base="${f#2023_}"       # remove '2022_'
    mv "$f" "${base%.json}_2023.json"
done
