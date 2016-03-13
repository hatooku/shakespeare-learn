import hmm_helper as base


# Data Visualization

# Find top words
def find_best(emiss, wordmap):
    states = len(emiss)
    best = {}
    for x in range(states):
        temp = list(emiss[x])
        for y in range(len(temp)):
            temp[y] = (temp[y], y)
        temp = sorted(temp)
        words = temp[::-1][0:10]
        for y in range(len(words)):
            words[y] = words[y][1]
        for y in range(len(words)):
            words[y] = wordmap[words[y]]
        best[x] = words
    return best

# Cleanly print top words
def clean_print_top(d):
    for x in d:
        print x+1
        print "-"*40
        temp = d[x]
        for y in temp:
            print y
        print ""

# Find most transitioned to states
def trans_max(trans):
    res = []
    for row in trans:
        temp = list(row)
        hi = max(temp)
        s = temp.index(hi)
        res.append((s, hi))
    return res

# Cleanly print highest transitions
def clean_print_trans(ans):
    print "Top Transitions"
    print "-"*40
    for x in range(len(ans)):
        print "State %d --> %d, prob %0.3f" %(x+1, ans[x][0], ans[x][1])


trans, emiss, wordmap, init = base.load_model("10-state-quatrains")
d = find_best(emiss, wordmap)
# clean_print_top(d)

ans = trans_max(trans)
clean_print_trans(ans)
