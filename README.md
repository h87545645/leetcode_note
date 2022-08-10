# LEET CODE STUDY NOTE

## 2022/8/10

## 640. 求解方程

[640. 求解方程](https://leetcode.cn/problems/solve-the-equation/)

```
640. 求解方程
求解一个给定的方程，将x以字符串 "x=#value" 的形式返回。该方程仅包含 '+' ， '-' 操作，变量 x 和其对应系数。

如果方程没有解，请返回 "No solution" 。如果方程有无限解，则返回 “Infinite solutions” 。

题目保证，如果方程中只有一个解，返回值 'x' 是一个整数。

 

示例 1：

输入: equation = "x+5-3+x=6+x-2"
输出: "x=2"
示例 2:

输入: equation = "x=x"
输出: "Infinite solutions"
示例 3:

输入: equation = "2x=x"
输出: "x=0"
```

`思路`
模拟正常解方程，计数所有x `xCnt`，计算所有数字和`res`,方程=号左边的数字取负，=号右边的x取负，最后如果`xCnt`和`res`都为0，则无限解。`xCnt`为0`res`不为0则无解，返回`res/xCnt`

`c# 实现`
```
public class Solution {
    public string SolveEquation(string equation) {

        int res = 0;
        int xCnt = 0;
        int flag = -1;

        for (int i = 0; i < equation.Length; i++)
        {
          
            if (Char.IsDigit(equation[i]) || equation[i] == 'x')
            {   
                int tempFlag = 1;
                if (i > 0)
                {
                    if (equation[i - 1] == '-')
                    {
                        tempFlag = -1;
                    }
                }
                int t = equation[i] == 'x' ? 1 : 0;
                while(i < equation.Length && char.IsDigit(equation[i])){
                    t = t * 10 + equation[i] - '0';
                    i++;
                }
                if (i < equation.Length)
                {
                    if (equation[i] == 'x')
                    {
                        xCnt += t * tempFlag * -flag;
                    }else{
                        res += t * tempFlag * flag;
                    }
                }else{
                    res += t * tempFlag * flag;
                }
            }
            if (i < equation.Length && equation[i] == '=')
            {
                flag = 1;
            }
        }
        if (xCnt == 0 && res == 0)
        {
            return "Infinite solutions";
        }
        if (xCnt == 0 && res != 0)
        {
            return "No solution";
        }
        res /= xCnt;
        return "x="+res;
    }
}
```

***


## 2022/8/9

## 1413. 逐步求和得到正数的最小值

[1413. 逐步求和得到正数的最小值](https://leetcode.cn/problems/minimum-value-to-get-positive-step-by-step-sum)

```
给你一个整数数组 nums 。你可以选定任意的 正数 startValue 作为初始值。

你需要从左到右遍历 nums 数组，并将 startValue 依次累加上 nums 数组中的值。

请你在确保累加和始终大于等于 1 的前提下，选出一个最小的 正数 作为 startValue 。

 

示例 1：

输入：nums = [-3,2,-3,4,2]
输出：5
解释：如果你选择 startValue = 4，在第三次累加时，和小于 1 。
                累加求和
                startValue = 4 | startValue = 5 | nums
                  (4 -3 ) = 1  | (5 -3 ) = 2    |  -3
                  (1 +2 ) = 3  | (2 +2 ) = 4    |   2
                  (3 -3 ) = 0  | (4 -3 ) = 1    |  -3
                  (0 +4 ) = 4  | (1 +4 ) = 5    |   4
                  (4 +2 ) = 6  | (5 +2 ) = 7    |   2
示例 2：

输入：nums = [1,2]
输出：1
解释：最小的 startValue 需要是正数。
示例 3：

输入：nums = [1,-2,-3]
输出：5

```

`思路`
只需保证累加的`sum`+`ans`>=1 即可，先计算出累加过程中最小的值，即可知道`ans`


`c# 实现`
```
public class Solution {
    public int MinStartValue(int[] nums) {
        int ans = 1 , sum = 0;
        for (int i = 0; i < nums.Length; i++)
        {
            sum += nums[i];
            ans = Math.Min(ans,sum);
        }
        if (ans < 1)
        {
            ans = 1 - ans;
        }
        return ans;
    }
}
```

***

## 2022/8/8

## 761. 特殊的二进制序列

[761. 特殊的二进制序列](https://leetcode.cn/problems/special-binary-string)

```
特殊的二进制序列是具有以下两个性质的二进制序列：

0 的数量与 1 的数量相等。
二进制序列的每一个前缀码中 1 的数量要大于等于 0 的数量。
给定一个特殊的二进制序列 S，以字符串形式表示。定义一个操作 为首先选择 S 的两个连续且非空的特殊的子串，然后将它们交换。（两个子串为连续的当且仅当第一个子串的最后一个字符恰好为第二个子串的第一个字符的前一个字符。)

在任意次数的操作之后，交换后的字符串按照字典序排列的最大的结果是什么？

示例 1:

输入: S = "11011000"
输出: "11100100"
解释:
将子串 "10" （在S[1]出现） 和 "1100" （在S[3]出现）进行交换。
这是在进行若干次操作后按字典序排列最大的结果。
```

`思路`
这道题目理解起来比较难，看了评论说其实就是找匹配的1和0 把1多的往前放，遍历字符串，当1和0数量相同时则找到了一个匹配的子串，将其放入list中，其字串也要执行相同的操作，最后排序所有匹配的1和0。

`c# 实现`
```
public class Solution {
    public string MakeLargestSpecial(string s) {
        if (s.Length <= 2)
        {
            return s;
        }
        List<String> list = new List<String>();
        int count = 0, prev = 0;
        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == '1')
            {
                ++count;
            }else{
                --count;
                if (count == 0)
                {
                    list.Add("1"+MakeLargestSpecial(s.Substring(prev+1 , i - prev - 1 ))+"0");
                    prev = i + 1;
                }
            }
        }
        list.Sort((a, b) => b.CompareTo(a));
        StringBuilder ans = new StringBuilder();
        foreach (string sub in list) {
            ans.Append(sub);
        }
        return ans.ToString();
    }

}
```

***

## 2022/8/5

## 623. 在二叉树中增加一行

[623. 在二叉树中增加一行. 非递增顺序的最小子序列](https://leetcode.cn/problems/add-one-row-to-tree)

```
给定一个二叉树的根 root 和两个整数 val 和 depth ，在给定的深度 depth 处添加一个值为 val 的节点行。

注意，根节点 root 位于深度 1 。

加法规则如下:

给定整数 depth，对于深度为 depth - 1 的每个非空树节点 cur ，创建两个值为 val 的树节点作为 cur 的左子树根和右子树根。
cur 原来的左子树应该是新的左子树根的左子树。
cur 原来的右子树应该是新的右子树根的右子树。
如果 depth == 1 意味着 depth - 1 根本没有深度，那么创建一个树节点，值 val 作为整个原始树的新根，而原始树就是新根的左子树。
 

示例 1:



输入: root = [4,2,6,3,1,5], val = 1, depth = 2
输出: [4,1,1,2,null,null,6,3,1,5]
示例 2:



输入: root = [4,2,null,3,1], val = 1, depth = 3
输出:  [4,2,null,1,1,3,null,null,1]

```

`思路`
二叉树的遍历应用，[BFS遍历](https://github.com/h87545645/Blog/blob/main/algorithm/DFS%26BFS.md)二叉树，记录当前深度`curDeep`,当`curDeep == depth - 1`时，创建两个新TreeNode，其左右分别是原节点的左右节点，并将原节点的左右指向这两个新节点。

`c# 实现`
```
public class Solution {
    public TreeNode AddOneRow(TreeNode root, int val, int depth) {
        if (depth == 1)
        {
            TreeNode newRoot = new TreeNode(val,root);
            return newRoot;
        }else{
            Queue nodeQue = new Queue();
            nodeQue.Enqueue(root);
            nodeQue.Enqueue(null);
            int curDeep = 1;
            while(nodeQue.Count > 0){
                TreeNode curNode = nodeQue.Dequeue() as TreeNode;
                if (curNode == null)
                {
                    if (nodeQue.Count == 0)
                    {
                        break;
                    }
                    nodeQue.Enqueue(null);
                    curDeep ++;
                    continue;
                }
                if (curDeep == depth - 1)
                {
                    TreeNode insetLeft = new TreeNode(val,curNode.left);
                    TreeNode insetRight = new TreeNode(val,null,curNode.right);
                    curNode.left = insetLeft;
                    curNode.right = insetRight;
                }else if(curDeep > depth - 1){
                    break;
                }
                if (curNode.left != null)
                {
                     nodeQue.Enqueue(curNode.left);
                }
                if (curNode.right != null)
                {
                     nodeQue.Enqueue(curNode.right);
                }
            }
            return root;
        }
    }
}
```

***

## 2022/8/4

## 1403. 非递增顺序的最小子序列

[原题地址  1403. 非递增顺序的最小子序列](https://leetcode.cn/problems/minimum-subsequence-in-non-increasing-order/)

```
给你一个数组 nums，请你从中抽取一个子序列，满足该子序列的元素之和 严格 大于未包含在该子序列中的各元素之和。

如果存在多个解决方案，只需返回 长度最小 的子序列。如果仍然有多个解决方案，则返回 元素之和最大 的子序列。

与子数组不同的地方在于，「数组的子序列」不强调元素在原数组中的连续性，也就是说，它可以通过从数组中分离一些（也可能不分离）元素得到。

注意，题目数据保证满足所有约束条件的解决方案是 唯一 的。同时，返回的答案应当按 非递增顺序 排列。

 

示例 1：

输入：nums = [4,3,10,9,8]
输出：[10,9] 
解释：子序列 [10,9] 和 [10,8] 是最小的、满足元素之和大于其他各元素之和的子序列。但是 [10,9] 的元素之和最大。 
示例 2：

输入：nums = [4,4,7,6,7]
输出：[7,7,6] 
解释：子序列 [7,7] 的和为 14 ，不严格大于剩下的其他元素之和（14 = 4 + 4 + 6）。因此，[7,6,7] 是满足题意的最小子序列。注意，元素按非递增顺序返回。  
示例 3：

输入：nums = [6]
输出：[6]

```

`思路`
对`nums`排序并求和`sum`，倒叙遍历`nums`，累加每次遍历的和`ansSum`，累减`sum`，并添加到返回的list中，当`ansSum`大于`sum`时，返回结果，

`c# 实现`
```
public class Solution {
    public IList<int> MinSubsequence(int[] nums) {
        IList<int> ans = new List<int>();
        Array.Sort(nums);
        int sum = 0,ansSum = 0;
        sum = nums.Sum();
        // for (int i = 0; i < nums.Length; i++)
        // {
        //     sum += nums[i];
        // }
        for (int i = nums.Length - 1; i >= 0; i--)
        {
            ans.Add(nums[i]);
            ansSum += nums[i];
            sum -= nums[i];
            if (ansSum > sum)
            {
                break;
            }
        }
        return ans;
    }
}
```

***

## 2022/8/3

## 899. 有序队列

[原题地址  899. 有序队列](https://leetcode.cn/problems/orderly-queue)

```
给定一个字符串 s 和一个整数 k 。你可以从 s 的前 k 个字母中选择一个，并把它加到字符串的末尾。

返回 在应用上述步骤的任意数量的移动后，字典上最小的字符串 。

 

示例 1：

输入：s = "cba", k = 1
输出："acb"
解释：
在第一步中，我们将第一个字符（“c”）移动到最后，获得字符串 “bac”。
在第二步中，我们将第一个字符（“b”）移动到最后，获得最终结果 “acb”。
示例 2：

输入：s = "baaca", k = 3
输出："aaabc"
解释：
在第一步中，我们将第一个字符（“b”）移动到最后，获得字符串 “aacab”。
在第二步中，我们将第三个字符（“c”）移动到最后，获得最终结果 “aaabc”。

```

`思路`
当k大于1时，一定可以将字符串换成从小到大的顺序，而k为1时，遍历string,记录每一次的交换结果，返回最小的那个

`c# 实现`
```
public class Solution {
    public string OrderlyQueue(string s, int k) {
        if (k == 1)
        {
            string shortS = s;
            StringBuilder sbuild = new StringBuilder(s);
            int n = s.Length;
            for (int i = 1; i < n; i++)
            {
                char head = sbuild[0];
                sbuild.Remove(0,1);
                sbuild.Append(head);
                if (sbuild.ToString().CompareTo(shortS) < 0)
                {
                    shortS =  sbuild.ToString();
                }
            }
            return shortS;
        }else{
            char[] sa = s.ToCharArray();
            Array.Sort(sa);
            return new string(sa);
        }
        
    }
}
```

***

## 2022/8/2

## 622. 设计循环队列

[原题地址  622. 设计循环队列](https://leetcode.cn/problems/design-circular-queue)

```
设计你的循环队列实现。 循环队列是一种线性数据结构，其操作表现基于 FIFO（先进先出）原则并且队尾被连接在队首之后以形成一个循环。它也被称为“环形缓冲器”。

循环队列的一个好处是我们可以利用这个队列之前用过的空间。在一个普通队列里，一旦一个队列满了，我们就不能插入下一个元素，即使在队列前面仍有空间。但是使用循环队列，我们能使用这些空间去存储新的值。

你的实现应该支持如下操作：

MyCircularQueue(k): 构造器，设置队列长度为 k 。
Front: 从队首获取元素。如果队列为空，返回 -1 。
Rear: 获取队尾元素。如果队列为空，返回 -1 。
enQueue(value): 向循环队列插入一个元素。如果成功插入则返回真。
deQueue(): 从循环队列中删除一个元素。如果成功删除则返回真。
isEmpty(): 检查循环队列是否为空。
isFull(): 检查循环队列是否已满。
 

示例：

MyCircularQueue circularQueue = new MyCircularQueue(3); // 设置长度为 3
circularQueue.enQueue(1);  // 返回 true
circularQueue.enQueue(2);  // 返回 true
circularQueue.enQueue(3);  // 返回 true
circularQueue.enQueue(4);  // 返回 false，队列已满
circularQueue.Rear();  // 返回 3
circularQueue.isFull();  // 返回 true
circularQueue.deQueue();  // 返回 true
circularQueue.enQueue(4);  // 返回 true
circularQueue.Rear();  // 返回 4

```

`思路`

链表实现

`c# 实现`
```
public class MyCircularQueue {
    private Node head;
    private Node tail;
    private int capacity;
    private int size;
    public MyCircularQueue(int k) {
        capacity = k;
        size = 0;
    }
    
    public bool EnQueue(int value) {
        if (IsFull()) {
            return false;
        }
        Node temp = new Node(value);
        if (size == 0)
        {
            head = temp;
            tail = temp;
        }else{
            // temp.next = head;
            tail.next = temp;
            tail = temp;
        }
        size ++;
        return true;
    }
    
    public bool DeQueue() {
        if(IsEmpty()){
            return false;
        }
        // tail.next = head.next;
        head = head.next;
        size --;
        return true;
    }
    
    public int Front() {
        if (IsEmpty()) {
            return -1;
        }
        return head.val;
    }
    
    public int Rear() {
        if (IsEmpty()) {
            return -1;
        }
        return tail.val;
    }
    
    public bool IsEmpty() {
        return size == 0;
    }
    
    public bool IsFull() {
        return size == capacity;
    }

    class Node {
        public int val;
        public Node next;
        public Node(int val, Node next = null){
            this.val = val;
            this.next = next;
        }
    }
}
```

***

## 2022/8/1

## 1374. 生成每种字符都是奇数个的字符串

[原题地址 1374. 生成每种字符都是奇数个的字符串](https://leetcode.cn/problems/generate-a-string-with-characters-that-have-odd-counts)
```
给你一个整数 n，请你返回一个含 n 个字符的字符串，其中每种字符在该字符串中都恰好出现 奇数次 。

返回的字符串必须只含小写英文字母。如果存在多个满足题目要求的字符串，则返回其中任意一个即可。

 

示例 1：

输入：n = 4
输出："pppz"
解释："pppz" 是一个满足题目要求的字符串，因为 'p' 出现 3 次，且 'z' 出现 1 次。当然，还有很多其他字符串也满足题目要求，比如："ohhh" 和 "love"。
示例 2：

输入：n = 2
输出："xy"
解释："xy" 是一个满足题目要求的字符串，因为 'x' 和 'y' 各出现 1 次。当然，还有很多其他字符串也满足题目要求，比如："ag" 和 "ur"。
示例 3：

输入：n = 7
输出："holasss"

```

`思路`
判断n的奇偶，返回n个字符或者1个字符+n-1个另外字符

`c# 实现`
```
public class Solution {
    public string GenerateTheString(int n) {
        StringBuilder ansBuilder = new StringBuilder();
        int off = 0;

        if ((n&1) == 0)
        {
            off = 1;
            ansBuilder.Append('z');
        }
        for (int i = 0; i < n - off; i++)
        {
            ansBuilder.Append('x');
        }
        return ansBuilder.ToString();
    }
}
```

***

## 2022/7/29

## 593. 有效的正方形

[原题地址 593. 有效的正方形](https://leetcode.cn/problems/valid-square)
```
给定2D空间中四个点的坐标 p1, p2, p3 和 p4，如果这四个点构成一个正方形，则返回 true 。

点的坐标 pi 表示为 [xi, yi] 。输入 不是 按任何顺序给出的。

一个 有效的正方形 有四条等边和四个等角(90度角)。

 

示例 1:

输入: p1 = [0,0], p2 = [1,1], p3 = [1,0], p4 = [0,1]
输出: True
示例 2:

输入：p1 = [0,0], p2 = [1,1], p3 = [1,0], p4 = [0,12]
输出：false
示例 3:

输入：p1 = [1,0], p2 = [-1,0], p3 = [0,1], p4 = [0,-1]
输出：true
```

`思路`
先对四个点进行排序，方便求边长，以左下和右上两点计算，两个边长必须大于0且相等，再计算两边的夹角，我用的勾股定理来判断是否为直角三角形，都满足则返回true

`c# 实现`
```
public class Solution {
    public bool ValidSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
        int [][] pos = new int[][]{p1,p2,p3,p4};
        //对四个点排序，从左到右 从下到上
        Array.Sort(pos,new CompareMethod());
        //求两边长是否一样
        double dis1 = GetDistance(pos[0],pos[1]);
        double dis2 = GetDistance(pos[0],pos[2]);
        if(dis1 <= 0 || dis2 <= 0 || dis1 != dis2){
            return false;
        }
        double dis3 = GetDistance(pos[3],pos[2]);
        double dis4 = GetDistance(pos[3],pos[1]);
               if(dis3 <= 0 || dis4 <= 0 || dis3 != dis4){
            return false;
        }
        double dis5 = GetDistance(pos[1],pos[2]);
        bool angel = Math.Round(dis1*dis1 + dis2*dis2,2) ==  Math.Round(dis5*dis5,2);
        if (!angel)
        {
            return false;
        }
                // return true;
         angel = Math.Round(dis3*dis3 + dis4*dis4,2) ==  Math.Round(dis5*dis5,2);

        if (!angel)
        {
            return false;
        }
        return true;
    }

    private double GetDistance(int[] a, int[] b)
    {
        int x=Math.Abs(b[0]   -   a[0] );
        int y=Math.Abs(b[1]   -   a[1] );
        return Math.Sqrt(x*x+y*y);
    }

    public class CompareMethod : IComparer<int[]>  //继承IComparer<T>接口，T为要比较的元素的类型
    {                                             //类中类，也可以放在类外面
        public  int Compare(int[] x, int[] y)
        {
            if (x[0] == y[0])
            {
                return x[1] - y[1];
            }else{ 
                return x[0] - y[0];
            }
        }
    }
}
```

***

## 2022/7/28

## 1331. 数组序号转换

[原题地址 1331. 数组序号转换](https://leetcode.cn/problems/rank-transform-of-an-array)

```
给你一个整数数组 arr ，请你将数组中的每个元素替换为它们排序后的序号。

序号代表了一个元素有多大。序号编号的规则如下：

序号从 1 开始编号。
一个元素越大，那么序号越大。如果两个元素相等，那么它们的序号相同。
每个数字的序号都应该尽可能地小。
 

示例 1：

输入：arr = [40,10,20,30]
输出：[4,1,2,3]
解释：40 是最大的元素。 10 是最小的元素。 20 是第二小的数字。 30 是第三小的数字。
示例 2：

输入：arr = [100,100,100]
输出：[1,1,1]
解释：所有元素有相同的序号。
示例 3：

输入：arr = [37,12,28,9,100,56,80,5,12]
输出：[5,3,4,2,8,6,7,1,3]

```

`思路`
拷贝arr并排序，用一个不重复的集合记录每个元素对应的排序，遍历`arr`,将元素对应的排序赋值给`ans`数组

`c# 实现`

```
public class Solution {
    public int[] ArrayRankTransform(int[] arr) {
        int[] sortedArr = new int[arr.Length];
        Array.Copy(arr, 0, sortedArr, 0, arr.Length);
        Array.Sort(sortedArr);
        Dictionary<int, int> ranks = new Dictionary<int, int>();
        int[] ans = new int[arr.Length];
        foreach (int a in sortedArr) {
            ranks.TryAdd(a, ranks.Count + 1);
        }
        for (int i = 0; i < arr.Length; i++) {
            ans[i] = ranks[arr[i]];
        }
        return ans;
    }
}
```

***

## 2022/7/27

## 592. 分数加减运算

[原题地址 592. 分数加减运算](https://leetcode.cn/problems/fraction-addition-and-subtraction)

```
给定一个表示分数加减运算的字符串 expression ，你需要返回一个字符串形式的计算结果。 

这个结果应该是不可约分的分数，即最简分数。 如果最终结果是一个整数，例如 2，你需要将它转换成分数形式，其分母为 1。所以在上述例子中, 2 应该被转换为 2/1。

 

示例 1:

输入: expression = "-1/2+1/2"
输出: "0/1"
 示例 2:

输入: expression = "-1/2+1/2+1/3"
输出: "1/3"
示例 3:

输入: expression = "1/3-1/2"
输出: "-1/6"

```

`思路`
模拟运算，遍历`expression`，遇到运算符则将左右两边分数计算，并约分，记录结果，直到计算完成。

`c# 实现`
```
public class Solution {
    public string FractionAddition(string expression) {
        string ans = "";
        for(int i = 0; i < expression.Length; i++){
            if(i > 0){
                if(expression[i] == '+' || expression[i] == '-'){
                    
                    int nextOperator1 = expression.IndexOf('+',i+1,expression.Length - 1 -(i+1));
                    int nextOperator2 = expression.IndexOf('-',i+1,expression.Length - 1- (i+1));
                    int nextOperator;
                    if(nextOperator1 == -1 && nextOperator2 == -1){
                        nextOperator = expression.Length;
                    }else{
                        if(nextOperator1 == -1){
                            nextOperator = nextOperator2;
                        }
                        else if(nextOperator2 == -1){
                            nextOperator = nextOperator1;
                        }else{
                            nextOperator = Math.Min(nextOperator1,nextOperator2);
                        }
                    }
                    string fraction2 = expression.Substring(i + 1,nextOperator - i -1 );
                    string fraction1;
                    if(ans.Length > 0){
                        fraction1 = ans;
                    }else{
                        fraction1 = expression.Substring(0,i);
                    }
                    ans = Calculate(fraction1 , fraction2 , expression[i]);
                    
                }
            }
        }
        if(ans.Length == 0){
            ans = expression;
        }
        return ans;
    }

    private string Calculate(string fraction1 , string fraction2 , char opera){
        string[] fra1nums = fraction1.Split('/');
        string[] fra2nums = fraction2.Split('/');
        int denominator = Convert.ToInt32(fra1nums[1]) *  Convert.ToInt32(fra2nums[1]);
        int factor1 = 1, factor2 = 1;
        if(fra1nums[0][0] == '-'){
            factor1 = -1;
            fra1nums[0] = fra1nums[0].Substring(1,fra1nums[0].Length - 1 );
        }
        if(fra2nums[0][0] == '-'){
            factor2 = -1;
            fra2nums[0] = fra2nums[0].Substring(1,fra2nums[0].Length - 1 );
        }
        
        int numerator1 = Convert.ToInt32(fra1nums[0])*factor1*Convert.ToInt32(fra2nums[1]);
        int numerator2 = Convert.ToInt32(fra1nums[1])*factor2*Convert.ToInt32(fra2nums[0]);
        int numerator = opera  == '+'? numerator1 + numerator2 :  numerator1 - numerator2;
        if(numerator == 0){
            return "0/1";
        }
        int gcd = this.GCD(Math.Abs(denominator),Math.Abs(numerator));
        numerator /= gcd;
        denominator /= gcd;
        if(numerator < 0){
            return '-'+Math.Abs(numerator).ToString()+'/' + denominator.ToString();
        }
        return numerator.ToString() + '/' + denominator.ToString();
    }

    private int GCD(int a, int b)
    {
        if (a % b == 0) return b;
        return GCD(b, a % b);
    }
}
```
***

## 2022/7/26

## 1206. 设计跳表

[原题地址 1206. 设计跳表](https://leetcode.cn/problems/design-skiplist/)

```
不使用任何库函数，设计一个 跳表 。

跳表 是在 O(log(n)) 时间内完成增加、删除、搜索操作的数据结构。跳表相比于树堆与红黑树，其功能与性能相当，并且跳表的代码长度相较下更短，其设计思想与链表相似。

例如，一个跳表包含 [30, 40, 50, 60, 70, 90] ，然后增加 80、45 到跳表中，以下图的方式操作：


Artyom Kalinin [CC BY-SA 3.0], via Wikimedia Commons

跳表中有很多层，每一层是一个短的链表。在第一层的作用下，增加、删除和搜索操作的时间复杂度不超过 O(n)。跳表的每一个操作的平均时间复杂度是 O(log(n))，空间复杂度是 O(n)。

了解更多 : https://en.wikipedia.org/wiki/Skip_list

在本题中，你的设计应该要包含这些函数：

bool search(int target) : 返回target是否存在于跳表中。
void add(int num): 插入一个元素到跳表。
bool erase(int num): 在跳表中删除一个值，如果 num 不存在，直接返回false. 如果存在多个 num ，删除其中任意一个即可。
注意，跳表中可能存在多个相同的值，你的代码需要处理这种情况。

 

示例 1:

输入
["Skiplist", "add", "add", "add", "search", "add", "search", "erase", "erase", "search"]
[[], [1], [2], [3], [0], [4], [1], [0], [1], [1]]
输出
[null, null, null, null, false, null, true, false, true, false]

解释
Skiplist skiplist = new Skiplist();
skiplist.add(1);
skiplist.add(2);
skiplist.add(3);
skiplist.search(0);   // 返回 false
skiplist.add(4);
skiplist.search(1);   // 返回 true
skiplist.erase(0);    // 返回 false，0 不在跳表中
skiplist.erase(1);    // 返回 true
skiplist.search(1);   // 返回 false，1 已被擦除

```

`思路`
关于[跳表 skiplist](https://github.com/h87545645/Blog/blob/main/data-structure/%E8%B7%B3%E8%A1%A8%20SkipList.md)

`c# 实现`
```
public class Skiplist {
    const int MAX_LEVEL = 10;
    const double P_FACTOR = 0.25;
    private SkiplistNode head;
    private int level;
    private Random random;
    public Skiplist() {
        this.head = new SkiplistNode(-1,MAX_LEVEL);
        this.level = 0;
        this.random = new Random();
    }
    
    public bool Search(int target) {
        
        
        SkiplistNode cur = this.head;
        //从最大level往下搜索路径
        for (int i = level - 1; i >= 0; i--)
        {
            //找到每层小于且最接近 target 的元素 
            while(cur.forward[i] != null && cur.forward[i].val < target){
                cur = cur.forward[i];
            }
        }
        cur = cur.forward[0];
        if (cur!= null && cur.val == target)
        {
            return true;
        }
        return false;
    }
    
    public void Add(int num) {
        //update 用于记录搜索路径中所有的SkiplistNode
        SkiplistNode[] update = new SkiplistNode[MAX_LEVEL];
        Array.Fill(update, head);
        SkiplistNode cur = this.head;
        //从最大level往下搜索路径
        for (int i = level - 1; i >= 0; i--)
        {
            while(cur.forward[i] != null && cur.forward[i].val < num){
                cur = cur.forward[i];
            }
            update[i] = cur;
        }
        int lv = this.GetRandom();
        this.level = Math.Max(lv,this.level);
        SkiplistNode newNode = new SkiplistNode(num,lv);
        for(int i =0;i<lv;i++){
            newNode.forward[i] = update[i].forward[i];
            update[i].forward[i] = newNode;
        }
    }
    
    public bool Erase(int num) {
        //update 用于记录搜索路径中所有的SkiplistNode
        SkiplistNode[] update = new SkiplistNode[MAX_LEVEL];
        SkiplistNode cur = head;
        //从最大level往下搜索路径
        for (int i = this.level - 1; i >= 0; i--)
        {
            while(cur.forward[i] != null && cur.forward[i].val < num){
                cur = cur.forward[i];
            }
            update[i] = cur;
        }
        cur = cur.forward[0];
        if(cur == null || cur.val != num){
            return false;
        }
        for(int i = 0;i< level; i++){
            if(update[i].forward[i] != cur){
               break; 
            }
            update[i].forward[i] = cur.forward[i];
        }
        //有为Null的层 需要更新level
        while(level > 1 && head.forward[level - 1] == null){
            level --;
        }
        return true;
    }
    
    private int GetRandom(){
        int lv = 1;
        while(random.NextDouble() < P_FACTOR && lv < MAX_LEVEL){
            lv ++;
        }
        return lv;
    }

    class SkiplistNode{
        public int val;
        public SkiplistNode[] forward;
        public SkiplistNode(int val,int maxLevel){
            this.val = val;
            this.forward = new SkiplistNode[maxLevel];
        }
    }
}
```

***

## 2022/7/25

## 919. 完全二叉树插入器

[原题地址 919. 完全二叉树插入器](https://leetcode.cn/problems/complete-binary-tree-inserter/)

```
919. 完全二叉树插入器
完全二叉树 是每一层（除最后一层外）都是完全填充（即，节点数达到最大）的，并且所有的节点都尽可能地集中在左侧。

设计一种算法，将一个新节点插入到一个完整的二叉树中，并在插入后保持其完整。

实现 CBTInserter 类:

CBTInserter(TreeNode root) 使用头节点为 root 的给定树初始化该数据结构；
CBTInserter.insert(int v)  向树中插入一个值为 Node.val == val的新节点 TreeNode。使树保持完全二叉树的状态，并返回插入节点 TreeNode 的父节点的值；
CBTInserter.get_root() 将返回树的头节点。
 

示例 1：



输入
["CBTInserter", "insert", "insert", "get_root"]
[[[1, 2]], [3], [4], []]
输出
[null, 1, 2, [1, 2, 3, 4]]

解释
CBTInserter cBTInserter = new CBTInserter([1, 2]);
cBTInserter.insert(3);  // 返回 1
cBTInserter.insert(4);  // 返回 2
cBTInserter.get_root(); // 返回 [1, 2, 3, 4]
```

`思路`

这道题主要运用[二叉树的广度优先遍历BFS](https://github.com/h87545645/Blog/blob/main/algorithm/DFS%26BFS.md)，任意左或右节点为null的节点为待添加节点的父节点候选，定义Queue<TreeNode> nodeQueue 保存这些候选节点，Insert时将新节点作为队列首节点的左或右节点加入，加入后如果队列首节点左右都有节点了，则出队，并把新加的节点入队

`c# 实现`
```
public class CBTInserter {
    TreeNode myNode;
    Queue<TreeNode> nodeQueue;
    public CBTInserter(TreeNode root) {
        myNode = root;
        nodeQueue = new Queue<TreeNode>();
        Queue<TreeNode> temp = new Queue<TreeNode>();
        temp.Enqueue(myNode);
        while(temp.Count > 0){
            TreeNode nd = temp.Dequeue();
            if (nd.left == null || nd.right == null)
            {
                nodeQueue.Enqueue(nd);
            }
            if (nd.left != null)
            {
                 temp.Enqueue(nd.left);
            }
            if (nd.right != null)
            {
                 temp.Enqueue(nd.right);
            }
        }
    }
    
    public int Insert(int val) {
        TreeNode nd = nodeQueue.Peek();
        TreeNode child = new TreeNode(val);
        if (nd.left == null)
        {
            nd.left = child;
        }else{
            nd.right = child;
            nodeQueue.Dequeue();
        }
        nodeQueue.Enqueue(child);
        return nd.val;
    }
    
    public TreeNode Get_root() {
        return myNode;
    }
}
```

***

## 2022/7/22

## 757. 设置交集大小至少为2

[原题地址 757. 设置交集大小至少为2](https://leetcode.cn/problems/set-intersection-size-at-least-two/)

```
一个整数区间 [a, b]  ( a < b ) 代表着从 a 到 b 的所有连续整数，包括 a 和 b。

给你一组整数区间intervals，请找到一个最小的集合 S，使得 S 里的元素与区间intervals中的每一个整数区间都至少有2个元素相交。

输出这个最小集合S的大小。

示例 1:

输入: intervals = [[1, 3], [1, 4], [2, 5], [3, 5]]
输出: 3
解释:
考虑集合 S = {2, 3, 4}. S与intervals中的四个区间都有至少2个相交的元素。
且这是S最小的情况，故我们输出3。
示例 2:

输入: intervals = [[1, 2], [2, 3], [2, 4], [4, 5]]
输出: 5
解释:
最小的集合S = {1, 2, 3, 4, 5}.

```

`思路`
先对`intervals`按`intervals[0]`升序排序，如果相同按`intervals[1]`降序，定义 `List<int> areaTemp`记录所有区间的共同元素的集合，倒叙遍历`intervals`，按题目要求`intervals`中的区间至少要2个，如果`areaTemp`中有两个元素在`intervals[i]`区间中，则遍历下一个，否则将`intervals[i][0]`和`intervals[i][0]+1`加入`areaTemp` 

`c# 实现`

```
public class Solution {
    public int IntersectionSizeTwo(int[][] intervals) {
        Array.Sort(intervals,new CompareMethod());
        // return intervals[2][0];
        List<int> areaTemp = new List<int>();
        for (int i = intervals.Length - 1; i >= 0; i --)
        {
            int cnt = 2;
            if (areaTemp.Count > 0)
            {
                foreach (var item in areaTemp)
                {
                    if (item >= intervals[i][0] && item <= intervals[i][1])
                    {
                        cnt --;
                        if (cnt == 0)
                        {
                            break;
                        }
                    }
                
                }
              
            }
            if (cnt > 0)
            {
                for (int j = 0; j < cnt; j++)
                {
                   areaTemp.Add(intervals[i][0]+j);
                }
            }
        }
        // return areaTemp[1];
        return areaTemp.Count;
    }

    public class CompareMethod : IComparer<int[]>  //继承IComparer<T>接口，T为要比较的元素的类型
    {                                             //类中类，也可以放在类外面
        public  int Compare(int[] x, int[] y)
        {
            if (x[0] == y[0])
            {
                return y[1] - x[1];
            }else{ 
                return x[0] - y[0];
            }
        }
    }
}
```

***

## 2022/7/21

## 814. 二叉树剪枝

[原题地址 814. 二叉树剪枝](https://leetcode.cn/problems/binary-tree-pruning)

```
给你二叉树的根结点 root ，此外树的每个结点的值要么是 0 ，要么是 1 。

返回移除了所有不包含 1 的子树的原二叉树。

节点 node 的子树为 node 本身加上所有 node 的后代。

 

示例 1：


输入：root = [1,null,0,0,1]
输出：[1,null,0,null,1]
解释：
只有红色节点满足条件“所有不包含 1 的子树”。 右图为返回的答案。
示例 2：


输入：root = [1,0,1,0,0,0,1]
输出：[1,null,1,null,1]
示例 3：


输入：root = [1,1,0,1,1,0,1,0]
输出：[1,1,0,1,1,null,1]


```

`思路`

二叉树DFS应用，递归二叉树，节点为null或者左右节点都为null且`val`为0时，返回null,否则返回该节点

`c# 实现`

```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public TreeNode PruneTree(TreeNode root) {
        if (root == null)
        {
            return null;
        }
        root.left = PruneTree(root.left);
        root.right = PruneTree(root.right);
        if (root.left == null && root.right == null && root.val != 1)
        {
            return null;
        }
        return root;
    }
}
```

***

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
做题中发现了c#的int[][] grid其实不是二维数组， [c# int [][] 和 int[,] 的区别](https://github.com/h87545645/Blog/blob/main/data-structure/c%23%20int%20%5B%5D%5B%5D%20%E5%92%8C%20int%5B%2C%5D%20%E7%9A%84%E5%8C%BA%E5%88%AB)
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
