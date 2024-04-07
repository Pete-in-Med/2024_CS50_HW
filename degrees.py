import csv
import sys

from util import Node, StackFrontier, QueueFrontier

# Maps names to a set of corresponding person_ids
names = {}

# Maps person_ids to a dictionary of: name, birth, movies (a set of movie_ids)
people = {}

# Maps movie_ids to a dictionary of: title, year, stars (a set of person_ids)
movies = {}


def load_data(directory):
    """
    Load data from CSV files into memory.
    """
    # Load people
    with open(f"{directory}/people.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            #把csv裡面的東西讀進來，但movies的部分從缺，先用set()假設
            people[row["id"]] = {
                "name": row["name"],
                "birth": row["birth"],
                "movies": set()
            }
            #不在裡面代表是不曾出現過的名字，直接加；如果是曾經出現過的名字，那就加入原本的集合
            if row["name"].lower() not in names:
                names[row["name"].lower()] = {row["id"]}
            else:
                names[row["name"].lower()].add(row["id"])

    # Load movies
    with open(f"{directory}/movies.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies[row["id"]] = {
                "title": row["title"],
                "year": row["year"],
                "stars": set()
            }

    # Load stars
    with open(f"{directory}/stars.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        #把people裡面缺的movies補起來；把movie裡面缺的stars補起來
        for row in reader:
            try:
                people[row["person_id"]]["movies"].add(row["movie_id"])
                movies[row["movie_id"]]["stars"].add(row["person_id"])
            #捕獲error，因為我猜可能會有error，但它並不是真正錯誤，只是我沒有要採集的資料而已，所以用捕獲的而不中止程式碼
            except KeyError:
                pass


def main():
    if len(sys.argv) > 2:
        sys.exit("Usage: python degrees.py [directory]")
    directory = sys.argv[1] if len(sys.argv) == 2 else "large"

    # Load data from files into memory
    print("Loading data...")
    load_data(directory)
    print("Data loaded.")

    #把兩個人的名字輸入進來，如果沒有就再輸
    source = person_id_for_name(input("Name: "))
    if source is None:
        sys.exit("Person not found.")
    target = person_id_for_name(input("Name: "))
    if target is None:
        sys.exit("Person not found.")

    path = shortest_path(source, target)

    if path is None:
        print("Not connected.")
    else:
        degrees = len(path)
        print(f"{degrees} degrees of separation.")
        #因為shortest path那邊還沒遇到，不過(None, source)應該是原點到初始點的距離
        #然後再把新的距離加上去
        path = [(None, source)] + path
        #i從0開始逐步增加，把每個stars都看一遍
        #這個部分實在是看不太懂，不太懂dictionary中path的部分
        for i in range(degrees):
            person1 = people[path[i][1]]["name"]
            person2 = people[path[i + 1][1]]["name"]
            movie = movies[path[i + 1][0]]["title"]
            print(f"{i + 1}: {person1} and {person2} starred in {movie}")


def shortest_path(source, target):
    # 用來跟蹤每個節點的前驅節點
    predecessor = {}

    # 創建一個frontier來實現廣度優先搜索
    frontier = QueueFrontier()

    # 初始化frontier，將起始節點加入其中
    # ()中分別為start node, parent node, action
    frontier.add(Node(source, None, None))

    # 開始搜索
    # 如果frontier是empty代表搜索完成
    while not frontier.empty():
        # 從frontier中取出節點
        node = frontier.remove()

        # 如果當前節點是目標節點，則結束搜索
        if node.state == target:
            path = []
            #一直在迴圈當中搜索結點，直到parent node就是start node
            while node.parent is not None:
                #action是movie_id，state是stars_id
                path.append((node.action, node.state))
                #每次往前找一個
                node = node.parent
            path.reverse()
            return path

        # 擴展當前節點，並將相鄰節點加入佇列
        for movie_id, person_id in neighbors_for_person(node.state):
            if not frontier.contains_state(person_id):
                #這和前面一樣，()中分別為start node, parent node, action
                child = Node(person_id, node, movie_id)
                frontier.add(child)
                predecessor[person_id] = node

    # 如果無法找到路徑，返回None
    return None



def person_id_for_name(name):
    """
    Returns the IMDB id for a person's name,
    resolving ambiguities as needed.
    """
    person_ids = list(names.get(name.lower(), set()))
    if len(person_ids) == 0:
        return None
    elif len(person_ids) > 1:
        print(f"Which '{name}'?")
        for person_id in person_ids:
            # people是之前創建的字典，從裡面找尋這樣的person_id，看哪些同名的演員跑出來
            person = people[person_id]
            name = person["name"]
            birth = person["birth"]
            print(f"ID: {person_id}, Name: {name}, Birth: {birth}")
        #這邊使用try and except的好處是可以捕獲使用者輸入錯誤的時候，而程式不會崩潰
        try:
            person_id = input("Intended Person ID: ")
            if person_id in person_ids:
                return person_id
        except ValueError:
            pass
        return None
    else:
        return person_ids[0]


def neighbors_for_person(person_id):
    """
    Returns (movie_id, person_id) pairs for people
    who starred with a given person.
    """
    movie_ids = people[person_id]["movies"]
    neighbors = set()
    for movie_id in movie_ids:
        for person_id in movies[movie_id]["stars"]:
            neighbors.add((movie_id, person_id))
    return neighbors

# 確保這個檔案只有在執行的時候，也就是__name__被改成__main__的時候才運行
if __name__ == "__main__":
    main()
