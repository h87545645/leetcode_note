# LEET CODE STUDY NOTE

## 2022/7/20

## 1260. 二维网格迁移

[原题地址 1260. 二维网格迁移](https://leetcode.cn/problems/shift-2d-grid/)

```
给你一个 m 行 n 列的二维网格 grid 和一个整数 k。你需要将 grid 迁移 k 次。

每次「迁移」操作将会引发下述活动：

位于 grid[i][j] 的元素将会移动到 grid[i][j + 1]。
位于 grid[i][n - 1] 的元素将会移动到 grid[i + 1][0]。
位于 grid[m - 1][n - 1] 的元素将会移动到 grid[0][0]。
请你返回 k 次迁移操作后最终得到的 二维网格。

```
`思路`
题目要求其实就是将 `grid` 中每个元素向后移动 `k` 个位置，超过长度的循环到开头的位置。所以只需遍历一次`grid`，算出每个元素移动之后的下标位置，将计算好的下标位置的值添加到 `IList<IList<int>> ans`中，最后返回`ans`
`c# 实现`
```
public class Solution {
    public IList<IList<int>> ShiftGrid(int[][] grid, int k) {
    int m = grid.Length, n = grid[0].Length;
        IList<IList<int>> ans = new List<IList<int>>();
        // int[,] temp = new int[m,n];
        for(int i = 0; i < m; i++){
            IList<int> temp = new List<int>();
            for (int j = 0; j < n; j++)
            {
                // temp[i,j] = grid[i][j];
                int idx = i *  n + j;
                int preidx = (idx - k%(m*n) + n * m)%(m*n);
                int prei = (int)Math.Floor((double)(preidx / n));
                int prej = preidx % n;
                // grid[i][j] = grid[prei][prej];
                temp.Add(grid[prei][prej]);
            }
            ans.Add(temp);
        }
        return ans;
    }
}
```
做题中发现了c#的int[][] grid其实不是二维数组， [int [][] 和 int[,] 的区别]()
***

## 2022/7/19

## 731. 我的日程安排表 II

[原题地址 731. 我的日程安排表 II](https://leetcode.cn/problems/my-calendar-ii/)

相关问题 [729. 我的日程安排表 I](https://leetcode.cn/problems/my-calendar-i/)

```
实现一个 MyCalendar 类来存放你的日程安排。如果要添加的时间内不会导致三重预订时，则可以存储这个新的日程安排。

MyCalendar 有一个 book(int start, int end)方法。它意味着在 start 到 end 时间内增加一个日程安排，注意，这里的时间是半开区间，即 [start, end), 实数 x 的范围为，  start <= x < end。

当三个日程安排有一些时间上的交叉时（例如三个日程安排都在同一时间内），就会产生三重预订。

每次调用 MyCalendar.book方法时，如果可以将日程安排成功添加到日历中而不会导致三重预订，返回 true。否则，返回 false 并且不要将该日程安排添加到日历中。

请按照以下步骤调用MyCalendar 类: MyCalendar cal = new MyCalendar(); MyCalendar.book(start, end)

 

示例：

MyCalendar();
MyCalendar.book(10, 20); // returns true
MyCalendar.book(50, 60); // returns true
MyCalendar.book(10, 40); // returns true
MyCalendar.book(5, 15); // returns false
MyCalendar.book(5, 10); // returns true
MyCalendar.book(25, 55); // returns true
解释： 
前两个日程安排可以添加至日历中。 第三个日程安排会导致双重预订，但可以添加至日历中。
第四个日程安排活动（5,15）不能添加至日历中，因为它会导致三重预订。
第五个日程安排（5,10）可以添加至日历中，因为它未使用已经双重预订的时间10。
第六个日程安排（25,55）可以添加至日历中，因为时间 [25,40] 将和第三个日程安排双重预订；
时间 [40,50] 将单独预订，时间 [50,55）将和第二个日程安排双重预订。

```

`分析`
相比以前做过的的[我的日程安排表 I](https://leetcode.cn/problems/my-calendar-i/)，二增加了双重预定的要求，可以定义一个`calendar`存日程预定，定义另一个字典`repetCalendar`存预定了两次的部分，如果新book的日程与`repetCalendar`有重叠，则返回false,否则遍历`calendar`将重叠部分加入`repetCalendar`，接着更新`calendar`返回true

`c# 实现`

```
public class MyCalendarTwo {
    private Dictionary<int,int> calendar;
    private Dictionary<int,int> repetCalendar;
    public MyCalendarTwo() {
        calendar = new Dictionary<int,int>();
        repetCalendar = new Dictionary<int,int>();
    }
    
    public bool Book(int start, int end) {
        if(calendar.Count == 0){
            calendar.Add(start, end);
            return true;
        }
        if (repetCalendar.Count > 0)
        {   
            //判断是否已经重复预定了
            foreach( KeyValuePair<int, int> kvp in repetCalendar ){
                if(end > kvp.Key && start < kvp.Value ){
                    return false;
                }
            }
        }
        
        //与calendar比较，共同的部分加入repetCalendar 
        foreach( KeyValuePair<int, int> kvp in calendar ){
            if(end > kvp.Key && start < kvp.Value ){
                int tempend = Math.Min(kvp.Value,end);
                int tempstart = Math.Max(kvp.Key,start);
                repetCalendar.Add(tempstart , tempend);
            }
        }
        if(calendar.ContainsKey(start)){
            calendar[start] = Math.Max(calendar[start] , end);
        }else{
            calendar.Add(start, end);
        }
        return true;
    }

}
```


***
## 2022/7/18

## 749. 隔离病毒

[原题地址 749. 隔离病毒](https://leetcode.cn/problems/contain-virus/)

`分析`
遍历数组，遇到病毒将virusAreaCnt为key的VirusArea类计入virusAreaDict字典，并递归搜索周围的区域，遇0则VirusArea的哈希neighbors Add, wall计数加1，遇1则标记原数组为-virusAreaCnt。
遍历结束后如果virusAreaDict count <= 1 则返回结果，否则比较 找到neighbors最大的字典，将值设为1，其余字典还原为1 重复上述操作

`c# 实现`
```
public class Solution
{
    static int[][] dirs = {new int[]{-1, 0}, new int[]{1, 0}, new int[]{0, -1}, new int[]{0, 1}};
    private int virusAreaCnt ;
    private Dictionary<int, VirusArea> virusAreaDict;
    class VirusArea
    {
        public VirusArea(){}
        public HashSet<int> neighbors = new HashSet<int>();
        public int wall = 0;
    }

    public int ContainVirus(int[][] isInfected)
    {
        int ans = 0;
        int rowCnt = isInfected.Length, colCnt = isInfected[0].Length;
        while (true)
        {    
            virusAreaDict = new Dictionary<int, VirusArea>();
            virusAreaCnt = 0;
            for (int row = 0; row < rowCnt; row++)
            {
                for (int col = 0; col < colCnt; col++)
                {
                    if (isInfected[row][col] == 1)
                    {
                    
                        virusAreaCnt++;

                        if (!virusAreaDict.ContainsKey(virusAreaCnt))
                        {
                            virusAreaDict.Add(virusAreaCnt,new VirusArea());
      
                        }
                        Queue<Tuple<int, int>> queue = new Queue<Tuple<int, int>>();
                        queue.Enqueue(new Tuple<int, int>(row, col));
                        isInfected[row][col] = -virusAreaCnt;
                        while (queue.Count > 0) {
                            Tuple<int, int> tuple = queue.Dequeue();
                            int x = tuple.Item1, y = tuple.Item2;
                            for (int d = 0; d < 4; ++d) {
                                int nx = x + dirs[d][0], ny = y + dirs[d][1];
                                if (nx >= 0 && nx < rowCnt && ny >= 0 && ny < colCnt) {
                                    if (isInfected[nx][ny] == 1) {
                                        queue.Enqueue(new Tuple<int, int>(nx, ny));
                                        isInfected[nx][ny] = -virusAreaCnt;
                                    } else if (isInfected[nx][ny] == 0) {
                                        virusAreaDict[virusAreaCnt].wall++;
                                        int idx = nx * colCnt + (ny);
                                         virusAreaDict[virusAreaCnt].neighbors.Add(idx);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (virusAreaDict.Count == 0)
            {
                break;
            }
            int maxVirusAreaCnt = 0;
            int key = 1;
            foreach (KeyValuePair<int, VirusArea> kv in virusAreaDict)
            {
                if(maxVirusAreaCnt < kv.Value.neighbors.Count){
                    maxVirusAreaCnt = kv.Value.neighbors.Count;
                    key = kv.Key;
                }
                maxVirusAreaCnt = Math.Max(maxVirusAreaCnt,kv.Value.neighbors.Count);
            }
            ans += virusAreaDict[key].wall;
            if (virusAreaDict.Count == 1)
            {
                break;
            }
            for (int row = 0; row < rowCnt; row++)
            {
                for (int col = 0; col < colCnt; col++)
                {
                    if (isInfected[row][col] < 0)
                    {
                        if (isInfected[row][col] != -key)
                        {
                            isInfected[row][col] = 1;
                        }
                        else
                        {
                            isInfected[row][col] = 2;
                        }
                    }
                }
            }
            foreach (KeyValuePair<int, VirusArea> kv in virusAreaDict)
            {
                if (kv.Key != key)
                {
                    int cnt = 0;
                    foreach (int idx in kv.Value.neighbors)
                    {
                        int row = (int)Math.Floor((double)(idx / colCnt));
                        int col = idx % colCnt;
                      
                        isInfected[row][col] = 1;
                        cnt ++;
                    }
                }
            }
        }
        
        return ans;
    }

}
```

其中用到的[元组 Tuple](https://github.com/h87545645/Blog/blob/main/data-structure/c%23%20%E5%85%83%E7%BB%84%20Tuple.md)

***

## 2022/7/15

## 558. 四叉树交集

[原题地址 558. 四叉树交集](https://leetcode.cn/problems/logical-or-of-two-binary-grids-represented-as-quad-trees/)

`题目比较长就不复制了，这道题主要难在四叉树的理解，实现一个递归方法，比较quadTree1和quadTree2，只要其中有一个是叶子isLeaf，则根据条件返回其中一个node。如果两个都不是叶子，则递归四个字节点合成一个节点，新节点根据规则判断是否为叶子`

`c# 实现`
```
public class Solution {

    public Node Intersect(Node quadTree1, Node quadTree2) {
        
        if (quadTree1.isLeaf || quadTree2.isLeaf)
        {
             if (quadTree1.isLeaf && quadTree1.val)
            {
                return new Node(quadTree1.val,quadTree1.isLeaf);
            }else if(quadTree1.isLeaf && !quadTree1.val){
                return new Node(quadTree2.val,quadTree2.isLeaf,quadTree2.topLeft,quadTree2.topRight,quadTree2.bottomLeft,quadTree2.bottomRight);
            }
            else if (quadTree2.isLeaf && quadTree2.val)
            {
                return new Node(quadTree2.val,quadTree2.isLeaf);
            }else{
                return new Node(quadTree1.val,quadTree1.isLeaf,quadTree1.topLeft,quadTree1.topRight,quadTree1.bottomLeft,quadTree1.bottomRight);
            }
        }else
        {
            Node o1 = Intersect(quadTree1.topLeft, quadTree2.topLeft);
        Node o2 = Intersect(quadTree1.topRight, quadTree2.topRight);
        Node o3 = Intersect(quadTree1.bottomLeft, quadTree2.bottomLeft);
        Node o4 = Intersect(quadTree1.bottomRight, quadTree2.bottomRight);
        if (o1.isLeaf && o2.isLeaf && o3.isLeaf && o4.isLeaf && o1.val == o2.val && o1.val == o3.val && o1.val == o4.val) {
            return new Node(o1.val, true);
        }
        return new Node(true, false, o1, o2, o3, o4);
        }
    }
}
```

***

## 2022/7/14

## 745. 前缀和后缀搜索

[原题地址 745. 前缀和后缀搜索](https://leetcode.cn/problems/prefix-and-suffix-search/)

```
设计一个包含一些单词的特殊词典，并能够通过前缀和后缀来检索单词。

实现 WordFilter 类：

WordFilter(string[] words) 使用词典中的单词 words 初始化对象。
f(string pref, string suff) 返回词典中具有前缀 prefix 和后缀 suff 的单词的下标。如果存在不止一个满足要求的下标，返回其中 最大的下标 。如果不存在这样的单词，返回 -1 。
 

示例：

输入
["WordFilter", "f"]
[[["apple"]], ["a", "e"]]
输出
[null, 0]
解释
WordFilter wordFilter = new WordFilter(["apple"]);
wordFilter.f("a", "e"); // 返回 0 ，因为下标为 0 的单词：前缀 prefix = "a" 且 后缀 suff = "e" 。


```

`分析`

本来想用每个单词的前后缀拼接作为key来构造[字典树 Trie](https://github.com/h87545645/Blog/blob/main/data-structure/%E5%AD%97%E5%85%B8%E6%A0%91.md)的。后来看到官方题解里有更简洁的方法。

```
计算每个单词的前缀后缀组合可能性

预先计算出每个单词的前缀后缀组合可能性，用特殊符号连接，作为键，对应的最大下标作为值保存入哈希表。检索时，同样用特殊符号连接前后缀，在哈希表中进行搜索。

```

`c#实现`

```
public class WordFilter {
    Dictionary<string, int> dictionary;

    public WordFilter(string[] words) {
        dictionary = new Dictionary<string, int>();
        for (int i = words.Length - 1; i >= 0; i--) {
            string word = words[i];
            int m = word.Length;
            for (int prefixLength = 1; prefixLength <= m; prefixLength++) {
                for (int suffixLength = 1; suffixLength <= m; suffixLength++) {
                    dictionary.TryAdd(word.Substring(0, prefixLength) + "#" + word.Substring(m - suffixLength), i);
                }
            }
        }
    }

    public int F(string pref, string suff) {
        if (dictionary.ContainsKey(pref + "#" + suff)) {
            return dictionary[pref + "#" + suff];
        }
        return -1;
    }
}

```

***

## 2022/7/13
## 735. 行星碰撞

[原题地址 735. 行星碰撞](https://leetcode.cn/problems/asteroid-collision/)

```
给定一个整数数组 asteroids，表示在同一行的行星。

对于数组中的每一个元素，其绝对值表示行星的大小，正负表示行星的移动方向（正表示向右移动，负表示向左移动）。每一颗行星以相同的速度移动。

找出碰撞后剩下的所有行星。碰撞规则：两个行星相互碰撞，较小的行星会爆炸。如果两颗行星大小相同，则两颗行星都会爆炸。两颗移动方向相同的行星，永远不会发生碰撞。

 

示例 1：

输入：asteroids = [5,10,-5]
输出：[5,10]
解释：10 和 -5 碰撞后只剩下 10 。 5 和 10 永远不会发生碰撞。
示例 2：

输入：asteroids = [8,-8]
输出：[]
解释：8 和 -8 碰撞后，两者都发生爆炸。
示例 3：

输入：asteroids = [10,2,-5]
输出：[10]
解释：2 和 -5 发生碰撞后剩下 -5 。10 和 -5 发生碰撞后剩下 10 。

```

`分析`

遍历asteroids，int i 指向头，int j 指向尾。每次循环 i j 下标的行星分别判断是否和相邻的相撞，相撞时删除对应行星。

`c# 实现`

```
public class Solution {
    public int[] AsteroidCollision(int[] asteroids) {
        List<int> aster = new List<int>(asteroids);
         for (int i = 0 , j = asteroids.Length - 1; i < asteroids.Length || j >= 0; i++ , j --)
            {
                if (i > 0 && i < asteroids.Length && asteroids[i] < 0 && asteroids[i - 1] > 0 )
                {
                    int sub = asteroids[i - 1] + asteroids[i];
                    // return new int[]{sub};
                    if (sub == 0)
                    {
                        asteroids = DeleteEle(asteroids , i - 1 , 2);
                        i -= 2;
                    }else if (sub > 0)
                    {
                        asteroids = DeleteEle(asteroids , i , 1);
                        i -= 1;
                    }else{
                        asteroids = DeleteEle(asteroids , i - 1 , 1);
                        i -= 1;
                    }
                }
                if (j < asteroids.Length - 1 && i >=0 && asteroids[j] > 0 && asteroids[j + 1] < 0 )
                {
                    int sub = asteroids[j + 1] + asteroids[j];
                    // return new int[]{sub};
                    if (sub == 0)
                    {
                        asteroids = DeleteEle(asteroids , j  , 2);
                        j += 2;
                    }else if (sub > 0)
                    {
                        asteroids = DeleteEle(asteroids , j + 1 , 1);
                        j += 1;
                    }else{
                        asteroids = DeleteEle(asteroids , j  , 1);
                        j += 1;
                    }
                }
            }
        return asteroids;
    }

    private int[] DeleteEle(int[] arrayBorn,int index,int Len) 
    {
        if (Len < 0)    //删除长度小于0，返回
        {
            return arrayBorn;
        }
        if ( (index + Len) > arrayBorn.Length)      //删除长度超出数组范围 
        {
            Len = arrayBorn.Length - index;         //将长度设置为能删除的最大长度
        }
        for (int i = 0;i < arrayBorn.Length - ( index + Len); i++)      //将删除元素后面的元素往前移动
        {
            if ((index + i + Len) > arrayBorn.Length)       //若删除元素+删除长度超过数组的范围，即无法从数组中找到移动元素，则用null替代
            {
                arrayBorn[index + i] = 0;
            }
            else            //若能用数组的元素替换则用数组中的元素
            {
                arrayBorn[index + i] = arrayBorn[index + i + Len];
            }           
        }
        /*不改变数组长度*/
        // for (int j =Len;j > 0; j--)         //将删除元素后多余的元素置为null值
        // {
        //     arrayBorn[arrayBorn.Length - j ] = null;
        // }
        int[] newArray = new int[arrayBorn.Length-Len];
        for (int j =0;j < newArray.Length;j++) 
        {
            newArray[j] = arrayBorn[j];
        }
        return newArray;
        // return arrayBorn;            
    }
}

```


***

## 2022/7/12
## 1252. 奇数值单元格的数目

```
给你一个 m x n 的矩阵，最开始的时候，每个单元格中的值都是 0。

另有一个二维索引数组 indices，indices[i] = [ri, ci] 指向矩阵中的某个位置，其中 ri 和 ci 分别表示指定的行和列（从 0 开始编号）。

对 indices[i] 所指向的每个位置，应同时执行下述增量操作：

ri 行上的所有单元格，加 1 。
ci 列上的所有单元格，加 1 。
给你 m、n 和 indices 。请你在执行完所有 indices 指定的增量操作后，返回矩阵中 奇数值单元格 的数目。

 

示例 1：



输入：m = 2, n = 3, indices = [[0,1],[1,1]]
输出：6
解释：最开始的矩阵是 [[0,0,0],[0,0,0]]。
第一次增量操作后得到 [[1,2,1],[0,1,0]]。
最后的矩阵是 [[1,3,1],[1,3,1]]，里面有 6 个奇数。
示例 2：



输入：m = 2, n = 2, indices = [[1,1],[0,0]]
输出：0
解释：最后的矩阵是 [[2,2],[2,2]]，里面没有奇数。
 

提示：

1 <= m, n <= 50
1 <= indices.length <= 100
0 <= ri < m
0 <= ci < n
 

进阶：你可以设计一个时间复杂度为 O(n + m + indices.length) 且仅用 O(n + m) 额外空间的算法来解决此问题吗？

```

[原题地址 1252. 奇数值单元格的数目](https://leetcode.cn/problems/cells-with-odd-values-in-a-matrix)

`分析`

根据题目要求，先遍历indices可以用两个数组分别记录行列增加的次数，再遍历矩阵，计算每个位置行列相加并且计数奇数。

`c#实现`

```
public class Solution {
    public int OddCells(int m, int n, int[][] indices) {
        int[] r = new int[m];
        int[] c = new int[n];
        int ans = 0;
       
        for(int i =0; i < indices.Length; i ++){
            r[indices[i][0]] ++;
            c[indices[i][1]] ++;
        }
        
        for (int i = 0; i < r.Length; i++)
        {
            for (int j = 0; j < c.Length; j++)
            {
                if (((r[i] + c[j]) & 1) == 1)
                {
                    ans ++;
                }
            }
        }
        return ans;
    }
}
```
关于[& 运算符](https://github.com/h87545645/Blog/blob/main/c%23/c%23%E7%BB%8F%E9%AA%8C%E6%80%BB%E7%BB%93.md#-%E8%BF%90%E7%AE%97%E7%AC%A6)

***

## 2022/7/11
## 676. 实现一个魔法字典
```
设计一个使用单词列表进行初始化的数据结构，单词列表中的单词 互不相同 。 如果给出一个单词，请判定能否只将这个单词中一个字母换成另一个字母，使得所形成的新单词存在于你构建的字典中。

实现 MagicDictionary 类：

MagicDictionary() 初始化对象
void buildDict(String[] dictionary) 使用字符串数组 dictionary 设定该数据结构，dictionary 中的字符串互不相同
bool search(String searchWord) 给定一个字符串 searchWord ，判定能否只将字符串中 一个 字母换成另一个字母，使得所形成的新字符串能够与字典中的任一字符串匹配。如果可以，返回 true ；否则，返回 false 。
示例：

输入
["MagicDictionary", "buildDict", "search", "search", "search", "search"]
[[], [["hello", "leetcode"]], ["hello"], ["hhllo"], ["hell"], ["leetcoded"]]
输出
[null, null, false, true, false, false]

解释
MagicDictionary magicDictionary = new MagicDictionary();
magicDictionary.buildDict(["hello", "leetcode"]);
magicDictionary.search("hello"); // 返回 False
magicDictionary.search("hhllo"); // 将第二个 'h' 替换为 'e' 可以匹配 "hello" ，所以返回 True
magicDictionary.search("hell"); // 返回 False
magicDictionary.search("leetcoded"); // 返回 False
```

[原题地址 676. 实现一个魔法字典](https://leetcode.cn/problems/implement-magic-dictionary/)

`分析`

读完题就感觉适合使用上周用过的 **[字典树 Trie](https://github.com/h87545645/Blog/blob/main/data-structure/%E5%AD%97%E5%85%B8%E6%A0%91.md)**  来实现。

在buildDict中添加字典树
```
class Trie{
    public bool IsFinished { get; set; }
    public Trie[] Child { get; set; }
    public Trie(){
        IsFinished = false;
        Child = new Trie[26];
    }
}
public void BuildDict(string[] dictionary) {
    foreach (string word in dictionary)
    {
        Trie cur = root;
        for (int i = 0; i < word.Length; i++)
        {
            int index = word[i] - 'a';
            if (cur.Child[index] == null)
            {
                cur.Child[index] = new Trie();
            }
            cur = cur.Child[index];
        }
        cur.IsFinished = true;
    }
}
```

Search里递归DFS方法，如果字典树中第一次找不到此字符，则用isModify标记修改了单词，第二次找不到单词或这在index == searchWord.Length时未被isModify，则返回false

```
 private bool DFS(string searchWord , int index ,Trie node, bool isModify){
    if (index == searchWord.Length)
    {
        return isModify && node.IsFinished;
    }
    int wordInex = searchWord[index] - 'a';
    if (node.Child[wordInex] != null)
    {
        if (DFS(searchWord , index + 1 , node.Child[wordInex] , isModify))
        {
             return true;
        }
    }
    if (!isModify)
    {
        for (int i = 0; i < 26; i++)
        {
            if (i != wordInex && node.Child[i] != null)
            {
                if (DFS(searchWord , index + 1 , node.Child[i] , true))
                {
                    return true;
                }
            }
        }
    }
    return false;
}
```

***

## 2022/7/8
## 1217. 玩筹码

>有 n 个筹码。第 i 个筹码的位置是 position[i] 。

>我们需要把所有筹码移到同一个位置。在一步中，我们可以将第 i 个筹码的位置从 position[i] 改变为:

>position[i] + 2 或 position[i] - 2 ，此时 cost = 0
>position[i] + 1 或 position[i] - 1 ，此时 cost = 1
>返回将所有筹码移动到同一位置上所需要的 最小代价 。

[原题地址 1217. 玩筹码](https://leetcode.cn/problems/minimum-cost-to-move-chips-to-the-same-position/)

`分析`

按照一般思路需要遍历postion,算出其他筹码移动到该下标的cost，最后取cost最小的一个。但这种方式要遍历两次position，时间复杂度为O(n的2次方)。此题多半不是考虑这种解法。观察cost规则，只有移动奇数位cost才会+1，可以用[贪心算法](https://github.com/h87545645/Blog/blob/main/algorithm/%E8%B4%AA%E5%BF%83%E7%AE%97%E6%B3%95.md)把偶数和奇数位筹码看作整体，结果只和最终位置是偶数位还是奇数位相关，所以只需遍历一次positon，记录奇(偶)数位置的个数，取小的返回就是最终结果。

`c# 实现`
```
public class Solution {
    public int MinCostToMoveChips(int[] position) {
        int even = 0, odd = 0;
        foreach (int pos in position) {
            if ((pos & 1) != 0) { //判断奇偶
                odd++;
            } else {
                even++;
            }
        }
        return Math.Min(odd, even);
    }
}

```

关于[& 运算符](https://github.com/h87545645/Blog/blob/main/c%23/c%23%E7%BB%8F%E9%AA%8C%E6%80%BB%E7%BB%93.md#-%E8%BF%90%E7%AE%97%E7%AC%A6)
​
 
***

## 2022/7/7
## 648. 单词替换

在英语中，我们有一个叫做 词根(root) 的概念，可以词根后面添加其他一些词组成另一个较长的单词——我们称这个词为 继承词(successor)。例如，词根an，跟随着单词 other(其他)，可以形成新的单词 another(另一个)。

现在，给定一个由许多词根组成的词典 dictionary 和一个用空格分隔单词形成的句子 sentence。你需要将句子中的所有继承词用词根替换掉。如果继承词有许多可以形成它的词根，则用最短的词根替换它。

你需要输出替换之后的句子。

```
示例 1：

输入：dictionary = ["cat","bat","rat"], sentence = "the cattle was rattled by the battery"
输出："the cat was rat by the bat"
示例 2：

输入：dictionary = ["a","b","c"], sentence = "aadsfasf absbs bbab cadsfafs"
输出："a a b c"
 

提示：

1 <= dictionary.length <= 1000
1 <= dictionary[i].length <= 100
dictionary[i] 仅由小写字母组成。
1 <= sentence.length <= 10^6
sentence 仅由小写字母和空格组成。
sentence 中单词的总量在范围 [1, 1000] 内。
sentence 中每个单词的长度在范围 [1, 1000] 内。
sentence 中单词之间由一个空格隔开。
sentence 没有前导或尾随空格。
```
`分析`

**由于要求用最短的词根替换，考虑对于sentence中的每个单词，由短至长遍历它所有的前缀。把dictionary 中所有词根放入哈希集合中，如果这个前缀出现在哈希集合中，则我们找到了当前单词的最短词根，将这个词根替换原来的单词。最后返回重新拼接的句子**

`c#实现`
```
public class Solution {
    public string ReplaceWords(IList<string> dictionary, string sentence) {
        ISet<string> dictionarySet = new HashSet<string>();
        foreach (string root in dictionary) {
            dictionarySet.Add(root);
        }
        string[] words = sentence.Split(" ");
        for (int i = 0; i < words.Length; i++) {
            string word = words[i];
            for (int j = 0; j < word.Length; j++) {
                if (dictionarySet.Contains(word.Substring(0, 1 + j))) {
                    words[i] = word.Substring(0, 1 + j);
                    break;
                }
            }
        }
        return String.Join(" ", words);
    }
}

```

`解法二 字典树`

**看到一种有意思的解法，`字典树` 关于字典树 [字典树](https://github.com/h87545645/Blog/blob/main/data-structure/%E5%AD%97%E5%85%B8%E6%A0%91.md)**

**我们用 dictionary 中所有词根构建一棵字典树，并用特殊符号标记结尾。在搜索前缀时，只需在字典树上搜索出一条最短的前缀路径即可。**

`c#实现`
```
public class Solution {
    public string ReplaceWords(IList<string> dictionary, string sentence) {
        Trie trie = new Trie();
        foreach (string word in dictionary) {
            Trie cur = trie;
            for (int i = 0; i < word.Length; i++) {
                char c = word[i];
                if (!cur.Children.ContainsKey(c)) {
                    cur.Children.Add(c, new Trie());
                }
                cur = cur.Children[c];
            }
            cur.Children.Add('#', new Trie());
        }
        string[] words = sentence.Split(" ");
        for (int i = 0; i < words.Length; i++) {
            words[i] = FindRoot(words[i], trie);
        }
        return string.Join(" ", words);
    }

    public string FindRoot(string word, Trie trie) {
        StringBuilder root = new StringBuilder();
        Trie cur = trie;
        for (int i = 0; i < word.Length; i++) {
            char c = word[i];
            if (cur.Children.ContainsKey('#')) {
                return root.ToString();
            }
            if (!cur.Children.ContainsKey(c)) {
                return word;
            }
            root.Append(c);
            cur = cur.Children[c];
        }
        return root.ToString();
    }
}

public class Trie {
    public Dictionary<char, Trie> Children;

    public Trie() {
        Children = new Dictionary<char, Trie>();
    }
}

```

***

## 2022/7/6
## 736. Lisp 语法解析
给你一个类似 Lisp 语句的字符串表达式 expression，求出其计算结果。

表达式语法如下所示:

表达式可以为整数，let 表达式，add 表达式，mult 表达式，或赋值的变量。表达式的结果总是一个整数。
(整数可以是正整数、负整数、0)
let 表达式采用 "(let v1 e1 v2 e2 ... vn en expr)" 的形式，其中 let 总是以字符串 "let"来表示，接下来会跟随一对或多对交替的变量和表达式，也就是说，第一个变量 v1被分配为表达式 e1 的值，第二个变量 v2 被分配为表达式 e2 的值，依次类推；最终 let 表达式的值为 expr表达式的值。
add 表达式表示为 "(add e1 e2)" ，其中 add 总是以字符串 "add" 来表示，该表达式总是包含两个表达式 e1、e2 ，最终结果是 e1 表达式的值与 e2 表达式的值之 和 。
mult 表达式表示为 "(mult e1 e2)" ，其中 mult 总是以字符串 "mult" 表示，该表达式总是包含两个表达式 e1、e2，最终结果是 e1 表达式的值与 e2 表达式的值之 积 。
在该题目中，变量名以小写字符开始，之后跟随 0 个或多个小写字符或数字。为了方便，"add" ，"let" ，"mult" 会被定义为 "关键字" ，不会用作变量名。
最后，要说一下作用域的概念。计算变量名所对应的表达式时，在计算上下文中，首先检查最内层作用域（按括号计），然后按顺序依次检查外部作用域。测试用例中每一个表达式都是合法的。有关作用域的更多详细信息，请参阅示例。


`示例：`
```
示例 1：

输入：expression = "(let x 2 (mult x (let x 3 y 4 (add x y))))"
输出：14
解释：
计算表达式 (add x y), 在检查变量 x 值时，
在变量的上下文中由最内层作用域依次向外检查。
首先找到 x = 3, 所以此处的 x 值是 3 。
示例 2：

输入：expression = "(let x 3 x 2 x)"
输出：2
解释：let 语句中的赋值运算按顺序处理即可。
示例 3：

输入：expression = "(let x 1 y 2 x (add x y) (add x y))"
输出：5
解释：
第一个 (add x y) 计算结果是 3，并且将此值赋给了 x 。 
第二个 (add x y) 计算结果是 3 + 2 = 5 。


```
`分析`
***每个表达式都包含在()里，且有let add mult 三种关键字，其余都是变量名和整数，可以实现一个EvaInt函数来返回当前下标得值并移动下标，实现一个EvaVar函数来返回当前下标的变量字符穿并移动下标，因为let的赋值可能有多个，变量需要在对应作用域下，可以申明一个Dictionary<string,Stack<int>> scope 来存对应的变量值。考虑实现一个递归函数InnerEva
返回表达式的结果，只要当前字符不为左括号’(’，则判断为变量或值，直接返回变量或整数，然后判断判断是let add 或 mult 。add返回两变量的和，mult返回两变量的积， let 需要赋值所有的变量直到遇到左右括号（）则递归（）里的表达式。***

`c#实现`
```
public class Solution {
    int index = 0;
    Dictionary<string,Stack<int>> scope = new Dictionary<string,Stack<int>>(); //用来记录作用域内所有变量的值
    public int Evaluate(string expression) {
        return InnerEva(expression);
    }
    
    private int InnerEva(string expression){
        //不是（ 则只可能是 变量或者值 返回值或者变量的值
        if (expression[index] != '(')
        {
            if (char.IsLower(expression[index]))
            {
                string var = EvaVar(expression);
                return scope[var].Peek();
            }else{ //整数
                return EvaInt(expression);
            }   
        }
        // 下面处理括号内的表达式
        //移除(
        int ret;
        index++;
        //判断是let add 或 mult
        if (expression[index] == 'l')
        {
            index += 4;
            IList<string> vars = new List<string>(); //记录所有的变量名
            while (true)
            {
                if (!char.IsLower(expression[index])) //如果不是变量的字符 这时应该是下一个表达式的（  此时直接递得到下一个（）的值
                {
                    ret = InnerEva(expression);
                    break;
                }
                //记录该scope let表达式里 所有变量的值
                string var = EvaVar(expression);
                if (expression[index] == ')') //如果let 表达式结束 则返回最后赋值的变量值
                {
                    ret = scope[var].Peek();
                    break;
                }
                vars.Add(var);
                index ++;
                int v = InnerEva(expression);
                if (!scope.ContainsKey(var))
                {
                    scope.Add(var,new Stack<int>());
                }
                scope[var].Push(v);
                index++;
            }
            foreach (string var in vars) {
                scope[var].Pop(); // 清除当前作用域的变量
            }
        }else if (expression[index] == 'a')
        {
            index += 4;
            int v1 = InnerEva(expression);
            index ++;
            int v2 = InnerEva(expression);
            ret = v1 + v2;
        }else
        {
            index += 5;
            int v1 = InnerEva(expression);
            index ++;
            int v2 = InnerEva(expression);
            ret = v1 * v2;
        }
        //移除 ）
        index ++;
        return ret;
    }

    //返回当前下表的值
    private int EvaInt(string expression){
        int n = expression.Length;
        int ret = 0, sign = 1;
        if(expression[index] == '-'){
            sign = -1;
            index ++;
        }
        while (index < n && char.IsDigit(expression[index]))
        {
            ret = ret * 10 + (expression[index] - '0');
            index ++;
        }
        return ret*sign;
    }

    //返回当前下表的变量字符
    private string EvaVar(string expression){
        int n = expression.Length;
        StringBuilder ret = new StringBuilder();
        while (index < n && expression[index] != ' ' && expression[index] != ')')
        {
            ret.Append(expression[index]);
            index ++;
        }
        return ret.ToString();
    }
}
```


***


## 2022/7/5
## 729. 我的日程安排表 I
实现一个 MyCalendar 类来存放你的日程安排。如果要添加的日程安排不会造成 重复预订 ，则可以存储这个新的日程安排。

当两个日程安排有一些时间上的交叉时（例如两个日程安排都在同一时间内），就会产生 重复预订 。

日程可以用一对整数 start 和 end 表示，这里的时间是半开区间，即 [start, end), 实数 x 的范围为，  start <= x < end 。

实现 MyCalendar 类：

MyCalendar() 初始化日历对象。
boolean book(int start, int end) 如果可以将日程安排成功添加到日历中而不会导致重复预订，返回 true 。否则，返回 false 并且不要将该日程安排添加到日历中。

`示例：`
```
输入：
["MyCalendar", "book", "book", "book"]
[[], [10, 20], [15, 25], [20, 30]]
输出：
[null, true, false, true]

解释：
MyCalendar myCalendar = new MyCalendar();
myCalendar.book(10, 20); // return True
myCalendar.book(15, 25); // return False ，这个日程安排不能添加到日历中，因为时间 15 已经被另一个日程安排预订了。
myCalendar.book(20, 30); // return True ，这个日程安排可以添加到日历中，因为第一个日程安排预订的每个时间都小于 20 ，且不包含时间 20 。

```
`分析`
***可以申明一个Dictionary<int,int> 来记录日历区间，key表示日程开始，vaule表示日程结束。每次book遍历calendar 判断start和end是否在已有区间则返回false。***

`c#实现`
```
public class MyCalendar {
    private Dictionary<int,int>calendar = new Dictionary<int,int>();
    public MyCalendar() {
        calendar.Clear();
    }
    
    public bool Book(int start, int end) {
        if(calendar.Count == 0){
            calendar.Add(start, end);
            return true;
        }
        
        //遍历遍历calendar
        //如果start比value还大或者end比key还小,则没有重合
        //否则重合
        // int minKey = (from d in calendar orderby d.Key ascending select d.Key).First();
        // int maxValue = (from d in calendar orderby d.Value ascending select d.Key).Last();
        foreach( KeyValuePair<int, int> kvp in calendar ){
            if(start > kvp.Key && start < kvp.Value ){
                return false;
            }else if(end > kvp.Key && end < kvp.Value){
                return false;
            }else if(start <= kvp.Key && end >= kvp.Value){
                return false;
            }
        }
        calendar.Add(start, end);
        return true;
    }
}
```


***

## 2022/7/4
## 1200. 最小绝对差
给你个整数数组 arr，其中每个元素都 不相同。
请你找到所有具有最小绝对差的元素对，并且按升序的顺序返回。
示例 1：
输入：arr = [4,2,1,3]
输出：[[1,2],[2,3],[3,4]]
示例 2：
输入：arr = [1,3,6,10,15]
输出：[[1,3]]
示例 3：
输入：arr = [3,8,-10,23,19,-4,-14,27]
输出：[[-14,-10],[19,23],[23,27]]
提示：
2 <= arr.length <= 10^5
-10^6 <= arr[i] <= 10^6
***
`分析`
***题目要求输出值升序排列，为避免找到结果后再排序，可以先对arr进行一次sort。定义最小差valmin,返回数组res，然后遍历一次arr 将当前下表i的值与i+1比较，如果差值比记录的小，则清空res并添加i和i+1 如果差值和记录一样，则添加i和i+1。最后返回res就是要求数组。***

`c#实现`
```
public class Solution {
    public IList<IList<int>> MinimumAbsDifference(int[] arr) {
        IList<IList<int>> res = new List<IList<int>>();
        int valMin = -1;
        int n = arr.Length;
        Array.Sort(arr);
        for(int i = 0; i < n; i++){
            if(i == n - 1)break;
                int val = Math.Abs(arr[i] - arr[i+1]);
                if(valMin < 0 || val < valMin){
                    valMin = val;
                    res.Clear();
                    List<int> temp = new  List<int>();
                    temp.Add(arr[i]);
                    temp.Add(arr[i + 1]);
 
                    res.Add(temp);
                }
                else if(val == valMin){
                    
                    List<int> temp = new  List<int>();
                    temp.Add(arr[i]);
                    temp.Add(arr[i + 1]);
  
                    res.Add(temp);
                }

        }
        return res;
    }
}
```
