depth2_elos_high="2000 2050 2100 2150 2200 2250 2300"
depth2_elos_low="1500 1550 1600 1650 1700 1750 1800"
depth3_elos_high="2200 2250 2300 2350 2400 2450 2500"
depth3_elos_low="1800 1850 1900 1950 2000 2050 2100"
depth4_elos_high="2700 2650 2600 2550 2500 2450"
depth4_elos_low="1950 2000 2050 2100 2150 2200"
depth5_elos_high="2700 2650 2600 2550 2500 2450"
depth5_elos_low="1950 2000 2050 2100 2150 2200"

gpu=0
for d in 2 3 4 5; do
  for w in 1 2 3 5; do
    # pick the right var name based on (d,w)
    if (( d == 2 && w < 3 )); then elos=$depth2_elos_low;
    elif (( d == 2 )); then elos=$depth2_elos_high;
    elif (( d == 3 && w < 3 )); then elos=$depth3_elos_low;
    elif (( d == 3 )); then elos=$depth3_elos_high;
    elif (( d == 4 && w < 3 )); then elos=$depth4_elos_low;
    elif (( d == 4 )); then elos=$depth4_elos_high;
    elif (( d == 5 && w < 3 )); then elos=$depth5_elos_low;
    else elos=$depth5_elos_high;
    fi

    for e in $elos; do
      echo "Launching depth=$d width=$w elo=$e on GPU $gpu"
      python relative_base.py $d $w $gpu $e > logs/d${d}_w${w}_elo${e}.log 2>&1 &
      gpu=$(( (gpu+1) % 8 ))
    done
  done
done

wait
