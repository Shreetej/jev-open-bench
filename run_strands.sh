set -x
SD=../sdenv-main/bin/strands-decider
waitfor() { until grep -q "ok $1 " dl5.log; do sleep 30; done; }
for spec in "E4B-gemma4:google/gemma-4-E4B-it"; do
  v=${spec%%:*}; base=${spec#*:}; repo=amazon/strands-decider-$v-v1-2610; arm=strands-$v
  waitfor $repo; waitfor $base
  $SD serve $repo --port 8020 --device mps > serve_$v.log 2>&1 &
  SP=$!
  until curl -s -m 2 localhost:8020/v1/systemone -H 'content-type: application/json' -d '{"state":"x","questions":{"q":{"type":"noul","instructions":"Is this a test?"}}}' | grep -q answers; do sleep 5; kill -0 $SP 2>/dev/null || { echo "server $v died"; break; }; done
  .venv/bin/python open_bench.py run --arm $arm --url http://127.0.0.1:8020 --model $arm --null-desc-as-name
  .venv/bin/python open_bench.py run --arm $arm --url http://127.0.0.1:8020 --model $arm --null-desc-as-name --items results/sample4k_items.json --log results/full_calls_strands.jsonl
  kill $SP; sleep 5
done
echo ALLDONE
