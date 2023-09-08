# DDRel: A new dataset for interpersonal relation classification in dyadic dialogues

This paper has been accepted by AAAI2021.

load the samples:

```python
import json
with open(path,'r',encoding = 'utf-8') as f:
    for line in f:
        sample = json.loads(line.strip())
```

Here is an example of a sample:
```
dict_keys(['pair-id', 'session-id', 'label', 'context', 'nameA', 'nameB'])
{'pair-id': '0', 'session-id': '0', 'label': '5', 'context': ['B: Here it is. Pray for me, Gallagher.', "A: Stew, your hands are shaking. You've been drinking again.", 'B: Come on, come on. Here they come, Gallagher!', 'A: The boss is getting hoarse.', "B: There's the third one. If I don't get the last one, there's a certain sob sister I know that's going to get a kick right in the . . . oh! Whoops, almost had that."], 'nameA': 'GALLAGHER', 'nameB': 'STEW'}
```

Correspondence between label numbers and relationship categories:

* 1	--	Child-Parent
* 2	--	Child-Other Family Elder
* 3	--	Siblings
* 4	--	Spouse
* 5	--	Lovers
* 6	--	Courtship
* 7	--	Friends
* 8	--	Neighbors
* 9	--	Roommates
* 10	--	Workplace Superior - Subordinate
* 11	--	Colleague/Partners
* 12	--	Opponents
* 13	--	Professional Contact


## 13-class classification

each relation type is a class

## 6-class classification
* Family Elder-Junior: 1,2
* Family Peer: 3,4
* Intimacy: 5,6
* Others Peer: 7,8,9
* Official Elder-Junior: 10
* Official Peer: 11,12,13

## 4-class classification
* Family: 1,2,3,4
* Intimacy: 5,6
* Others: 7,8,9
* Official: 10,11,12,13


 
