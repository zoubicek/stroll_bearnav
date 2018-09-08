#check arguments
case $# in
    1) echo "path prefix: '$1'" ;;
    *) echo "wrong number of argument! 1st: path to folder containing maps."  1>&2
        exit -1
        ;;
esac

mkdir "results"

f="$1/icra_2019_adamap/short"

if [ 1 == 1 ]
then
rosrun dynamic_reconfigure dynparam set /navigator "{'summaryMap': False, 'plasticMap': False,'histogramRating': False,'remapRotGain': 1.0}"&
rosparam set names_map  [M000,C001,C002,C003,C004,C005,C006,C007,C008,C009,C010,C011,C012]
rosparam set names_view [A001,A002,A003,A004,A005,A006,A007,A008,A009,A010,A011,A012]
roslaunch stroll_bearnav remapTest.launch folder_map:=$f folder_view:=$f
cp ~/.ros/Results.txt results/Map_rating.txt
fi

path=`pwd`
f="`pwd`/icra_2019_adamap"
cd $f
make
cd $path

echo "in `pwd`"

for i in Map_rating Map_adaptive Map_plastic Map_static;do grep reports results/$i.txt|awk '{a=$21-$23;b=(sqrt(a*a)+384)%768-384;print sqrt(b*b)}'| tee results/$i.err|sort -nr > results/$i.srt;done
echo MAP PLASTICITY TEST: Section 4.4
echo -ne "	Error of Adaptive VS Static: "
paste results/Map_adaptive.err results/Map_static.err 		|./icra_2019_adamap/t-test $confidence
echo -ne "	Error of Adaptive VS Plastic: "
paste results/Map_adaptive.err results/Map_plastic.err          |./icra_2019_adamap/t-test $confidence
echo -ne "	Error of Plastic VS Static: "
paste results/Map_plastic.err results/Map_static.err 		|./icra_2019_adamap/t-test $confidence
echo -ne "	Error of Adaptive VS Weighted: "
paste results/Map_plastic.err results/Map_rating.err 		|./icra_2019_adamap/t-test $confidence
echo
gnuplot icra_2019_adamap/map.gnu >results/map.fig