kill -- -1 && sleep 1s
git pull && sleep 2s
python3.6 run_style_transfer_service.py &
sleep 2s
python3.6 test_style_transfer_service.py
