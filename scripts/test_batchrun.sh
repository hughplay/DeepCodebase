# python batchrun.py scripts/test_batchrun.sh --quotas 5 4 2 2
# python batchrun.py scripts/test_batchrun.sh --quotas 4
# replace the command below with your own command, e.g. python train.py
echo "1. sleep 5s" \
    && sleep 5s

echo "2. sleep 7s" \
    && sleep 7s

echo "3. sleep 3s" && sleep 3s

echo "4. sleep 10s" && sleep 10s
