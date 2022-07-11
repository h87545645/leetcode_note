# LEET CODE STUDY NOTE

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
