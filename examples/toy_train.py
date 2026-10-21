"""Toy training command for `ideagen run`: reads a config as JSON on stdin and prints a noisy validation score.
Replace with your own script; it only has to read the JSON and print the metric."""
import json, math, random, sys
c = json.load(sys.stdin)
s = 70 - 4 * (math.log10(c['lr']) + 3) ** 2 + 3 * c['dropout'] * (c['layers'] >= 4) - 2 * (c['optimizer'] == 'sgd') * (c['lr'] <= 0.0003)
print(json.dumps({'val_acc': round(s + random.gauss(0, 0.5), 3)}))
