OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Initialize our own variables:
run=false
log=false

while getopts "rl" opt; do
    case "$opt" in
    r)  run=true
        ;;
    l)  log=true
        ;;
    esac
done

shift $((OPTIND-1))

[ "$1" = "--" ] && shift

start=`date +%s`

mkdir .build
g++ -I include -I /usr/include/eigen3 src/*.cpp -o .build/neural -std=c++1z -lboost_serialization
error=$?

end=`date +%s`
runtime=$((end-start))

if [ $error == 0 ]
then
	echo "Built Successfully in $((runtime))s"
	if $run
	then
		if $log
		then
			./run.sh > log
		else
			./run.sh
		fi
	fi
fi