set terminal fig color
set size 0.8,0.6
set key right bottom
set xlabel 'Registration error [px]'
set ylabel 'Probability [-]'
set ytics 0.2
plot [0:99] [:]\
'results/Fremen_0_Best_500_result.srt' using 1:(1-$0/1260) with lines title "Summary map with the most stable features" lw 1,\
'results/Fremen_1_Best_500_result.srt' using 1:(1-$0/1260) with lines title "Predictive map usin FreMEn order 1" lw 1,\
'results/Map_adaptive_LT.srt' using 1:(1-$0/1260) with lines title "Adaptive map" lw 1,\

#'results/Map_summary.srt' using 1:(1-$0/1260) with lines title "Summary map" lw 1,\
#'results/Fremen_2_Best_500_result.srt' using 1:(1-$0/1260) with lines title "FreMEn 2" lw 1,\
#'results/Fremen_0_Monte_Carlo_500_result.srt' using 1:(1-$0/1260) with lines title "FreMEnX 0" lw 1,\
#'results/Fremen_1_Monte_Carlo_500_result.srt' using 1:(1-$0/1260) with lines title "FreMEnX 1" lw 1,\
#'results/Fremen_2_Monte_Carlo_500_result.srt' using 1:(1-$0/1260) with lines title "FreMEnX 2" lw 1,\
