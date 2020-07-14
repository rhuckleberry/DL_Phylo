branch_stuff = set([":",".","0","1","2","3","4","5","6","7","8","9","e","-"])

def taxon_name_replace(tree):
    tree = tree.replace("taxon1", "A")
    tree = tree.replace("taxon2", "B")
    tree = tree.replace("taxon3", "C")
    tree = tree.replace("taxon4", "D")
    return tree

def getDeepestPair(tree,open_threshold=2):
    open_count = 0
    coma_count = 0
    A = ""
    B = ""
    for char in tree:
        if open_count == open_threshold:
            if char == ",":
                coma_count +=1
            elif char == ")":
                break
            elif char not in branch_stuff:
                if coma_count == 0:
                    A += char
                elif coma_count == 1:
                    B += char
        elif char == "(":
            open_count += 1
    return set((A,B))

def getClass(tree):
    #Double parathesis
    tree = taxon_name_replace(tree)
    for open_threshold in [2,3]:
        pair = getDeepestPair(tree,open_threshold=open_threshold)
        partition = {0: [set(("A","B")),set(("C","D"))],
                     2: [set(("A","D")),set(("B","C"))],
                     1: [set(("A","C")),set(("D","B"))]}
        for label,p in partition.items():
            if pair in p:
                return label
    print("uhhh\n\t",tree,"\n\t",pair)
    return None
