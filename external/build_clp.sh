mkdir coin
cd coin

wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
chmod u+x coinbrew
./coinbrew fetch Clp@master
./coinbrew build Clp
