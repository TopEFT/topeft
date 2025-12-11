for f in 2022EE_*.json; do
    base="${f#2022EE_}"       # remove '2022_'
    mv "$f" "${base%.json}_2022EE.json"
done
