import sqlite3
from sqlite3 import Error

# database = '/Users⁩/angel⁩/Documents⁩/projects⁩/models⁩/Model.db'
# conn = sqlite3.connect('/Documents⁩/projects⁩/models⁩/Model.db')
# c = conn.cursor()
anotation_found_flag = False
conn = sqlite3.connect('Model.db')
cur = conn.cursor()

# input: facial recognition id (string or int)
# output: path to annotation file (string), flag for result found or not found (bool)
def getAnnotationByFRId(encoding, id):
    rows = cur.execute("SELECT infoPath FROM faceInfo WHERE encoding=:ecoding and id=:id", {'ecoding': ecoding, 'id': id})
    rows = cur.fetchone()
    if row == '':
        anotation_found_flag = False
        print("path is empty")
    return row[0]

# input: name (string)
# output: none
# action: update path in db (string)
def updateAnnotationByFR(name,path):
    cur.execute("""UPDATE faceInfo SET infoPath=:path WHERE name=:name""", {'path' : path, 'name': name})
    return


# input: id (int)
# output: name (string)
def idToName(id):
    cur.execute("SELECT name FROM faceInfo WHERE id=:id", {'id': id})
    name = cur.fetchone()
    return name[0]

# input: name (string)
# output: all possible id (list of int)
def nameToId(name):
    cur.execute("SELECT id FROM faceInfo WHERE name=:name", {'name': name})
    ids = cur.fetchall()
    return ids

# input: name (string), path(string)
# output: none
# action:add new people to database
def addNewPeople(name,path):
    cur.execute("INSERT INTO faceInfo (name,picPath) VALUES (:name, :picPath)", {'name': name, 'picPath': path})
    return

#test
def getAnnotationByFRId1(id):
    cur.execute("SELECT infoPath FROM faceInfo WHERE id=:id", {'id': id})
    rows = cur.fetchone()
    if rows[0] == '' and rows[0] is not None:
        anotation_found_flag = False
        print("path is empty")
    else:
        anotation_found_flag = True
    print(rows[0])
    return rows[0]

def main():

    getAnnotationByFRId1(1)
    updateAnnotationByFR('Stella Tao','a')
    getAnnotationByFRId1(1)
    name = idToName(1)
    print(name)

    ids = nameToId('Stella Tao') 
    for id in ids:
        print(id)

    addNewPeople('Melissa Pan','123')
    name = idToName(2)
    print(name)
    ids = nameToId('Melissa Pan') 
    for id in ids:
        print(id)

    conn.commit()
    cur.close()
 
 
if __name__ == '__main__':
    main()

