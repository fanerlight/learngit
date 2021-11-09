#include<iostream>
#include<string>
#include<vector>
#include<deque>
#include<unordered_map>
#include<unordered_set>
#include<numeric>
#include<functional>
using std::vector;
using std::string;
using std::deque;

/*
board长度小于16，手里的小于5，目测可以用指数复杂度的方法完成。
祖玛游戏，用手上的球对board中的球做消除；手上的球可以任意选取。
对于状态压缩，不是二元变量有些不方便；暴力搜索？每个球可以选择n+1个位置，可以从m个球中任选一个，直到board为空
或者手里的球用完为止。递归实现，复杂度n^m，对于这个数据量来说可以完成？但是选择完球还要选择位置，又增加了一层循环
并且消除board中大于等于3的连续球个数也比较麻烦。动态规划还是贪心？果然还是枚举法来完成，使用图搜索算法。
回溯可以看作是dfs的图搜索，相对应的就是用迭代进行的bfs图搜索。可以进行剪枝操作降低具体的复杂度
*/
std::unordered_map<char,int> hash;

/*
状态的表示：board和hand向量可以构成状态，则状态应该是pair情况。比较状态可以用两个数组记录，或者两个Trie来记录
*/
class Trie{
private:
    Trie* children[5];
    bool is_end;
public:
    Trie(){
        std::fill(children,children+5,nullptr);
        is_end=false;
    }

    void insert(const vector<int>& seq){
        auto cur=this;
        for(auto& ele:seq){
            if(!(cur->children[ele]))
                cur->children[ele]=new Trie();
            cur=cur->children[ele];
        }
        cur->is_end=true;
    }

    bool query(const vector<int>& seq){
        auto cur=this;
        for(auto& ele:seq){
            if(cur->children[ele])
                cur=cur->children[ele];
            else
                return false;
        }
        return cur->is_end;
    }

};

class RYBGW{
private:
    std::unordered_set<int> storages[5];
public:
    RYBGW(){}

    void insert(const vector<int>& hand){
        for(int i=0;i<5;++i)
            storages[i].insert(hand[i]);
    }

    bool query(const vector<int>& hand){
        for(int i=0;i<5;++i){
            if(!storages[i].count(hand[i]))
                return false;
        }
        return true;
    }
};

//从第pos个位置开始看，怎样的消除办法比较好呢？用双指针从pos两边开始扩展
void eliminate(vector<int>& board_int,int pos){
    // if(board_int.size()==11)
    //     std::cout<<board_int[0];
    int left=pos,right=pos,last_left=pos,last_right=pos-1;
    while (left>=0&&right<board_int.size()&&board_int[left]==board_int[right])
    {
        while (left>0&&board_int[left-1]==board_int[right])
        {
            --left;
        }
        while (right<board_int.size()-1&&board_int[right+1]==board_int[left])
        {
            ++right;
        }
        if(right-left+1>=last_right-last_left+1+3){
            last_right=right;
            last_left=left;
            --left;
            ++right;
        }else{//应该是第一个中心都没有消除，可以直接return
            if(last_right<last_left)
                return;
            else
                break;
        }
    }
    //循环结束后符合的条件应该是至少会有一个部分被消除，left、right同时超限或者
    board_int.erase(board_int.begin()+last_left,board_int.begin()+last_right+1);
}

int findMinStep(string board,string hand){
    //字符到数字的映射
    {
        hash.insert({'R',0});
        hash.insert({'Y',1});
        hash.insert({'B',2});
        hash.insert({'G',3});
        hash.insert({'W',4});
    }
    //映射结果
    vector<int> board_int(board.size()),hand_int(5);
    for(int i=0;i<board.size();++i)
        board_int[i]=hash[board[i]];
    //board中没有出现的颜色，球可以直接抛弃
    for(int i=0;i<hand.size();++i)
        ++hand_int[hash[hand[i]]];
    //两种状态
    Trie* board_state=new Trie();
    board_state->insert(board_int);
    RYBGW* hand_state=new RYBGW();
    hand_state->insert(hand_int);// 5个G，所以是在3的位置上有5个。这个状态怎么记录？用用一个四层的unordered_set分别记录呗。
    typedef std::pair<vector<int>,vector<int>> States;
    deque<States> myque;
    myque.emplace_back(std::make_pair(board_int,hand_int));
    States cur;
    while (!myque.empty())
    {
        cur=myque.front();myque.pop_front();
        //状态的转移，需要通过外力还是状态自身的情况？状态自身包含两种情况——board排列和球的个数
        //显然，球的情况中顺序没关系，因此可以固定顺序来匹配；对于board排列，可以使用Trie数据结构
        //使用的球其实可以直接减就可以，如果是按层序搜索，每一层恰好少一个
        if(cur.first.empty())
            break;
        //开始使用当前状态和当前手中的球进行新状态的生成
        for(int i=0;i<5;++i){
            if(!cur.second[i])
                continue;
            vector<int> next_hand_int(cur.second);
            --next_hand_int[i];
            //插在开头，后面循环可以用剪枝
            {
                vector<int> next_board_int;next_board_int.reserve(cur.first.size()+1);
                next_board_int.insert(next_board_int.end(),i);
                next_board_int.insert(next_board_int.end(),cur.first.begin(),cur.first.end());
                eliminate(next_board_int,0);
                if(!(board_state->query(next_board_int)&&hand_state->query(next_hand_int))){
                    myque.emplace_back(std::make_pair(next_board_int,next_hand_int));
                    board_state->insert(next_board_int);
                    hand_state->insert(next_hand_int);
                }
            }
            //把当前这个球插入cur.first中，需要考量插入时左右的情况，如果破坏两个连续颜色，可以直接跳过
            //尽量插入到相同颜色旁边，且插入到两旁或者中间是一样的情况。如果是两边颜色都不同，则可以插入
            int j=0;
            while(j<cur.first.size()-1){
                //下一个插入的board状态，插在j的后面
                if(i==cur.first[j+1]){
                    ++j;
                    continue;
                }
                vector<int> next_board_int;next_board_int.reserve(cur.first.size()+1);
                int ball1=cur.first[j],ball2=cur.first[j+1];
                //插在后面
                next_board_int.insert(next_board_int.end(),cur.first.begin(),cur.first.begin()+j+1);
                next_board_int.insert(next_board_int.end(),i);
                next_board_int.insert(next_board_int.end(),cur.first.begin()+j+1,cur.first.end());
                //接下来就是消除时刻，把next_board_int连续的情况消除。因为新插入的是在j这个位置，所以建议从这个位置开始消除
                eliminate(next_board_int,j);
                if(board_state->query(next_board_int)&&hand_state->query(next_hand_int)){
                    ++j;
                    continue;
                }
                myque.emplace_back(std::make_pair(next_board_int,next_hand_int));
                board_state->insert(next_board_int);
                hand_state->insert(next_hand_int);
                ++j;
            }
            {//插在后面
                vector<int> next_board_int;next_board_int.reserve(cur.first.size()+1);
                next_board_int.insert(next_board_int.end(),cur.first.begin(),cur.first.end());
                next_board_int.insert(next_board_int.end(),i);
                eliminate(next_board_int,j);
                if(board_state->query(next_board_int)&&hand_state->query(next_hand_int))
                    continue;
                myque.emplace_back(std::make_pair(next_board_int,next_hand_int));
                board_state->insert(next_board_int);
                hand_state->insert(next_hand_int);
            }
        }
    }
    auto rest=std::accumulate(cur.second.begin(),cur.second.end(),0);
    //如果弹出的状态中，board确实为空，则说明消除成功，因此要返回花费的最少球
    return cur.first.empty()?(hand.size()-rest):-1;
}

int main(int argc,const char* argv[]){
    string hand("WB"),board("RRWWRRBBRR");//0044002200
    auto ans=findMinStep(board,hand);
    std::cout<<ans<<std::endl;
    return 0;
}