MODEL_NAME=$1
python3 dotnets.py | dot -Tpng > $MODEL_NAME.png
