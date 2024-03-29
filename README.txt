git add filename(s) : append the workspace files to the stage 
git commit -m "comment": transfer the stage files to the head/master
查看当前版本库中有多少提交，使用git log，可以得到历史提交的版本号
可以用git reset version_number 进行回溯或者前进
如果回到过去之后，想要回到未来，可用git reflog查看未来版本号
查看当前git版本库状态，使用git status，可以知道工作区、暂存区中的修改记录与提交情况
比较工作区和master中的版本区别，使用git diff version_number
如果对工作区的文件进行了错误的修改，但还未进行git add 将其添加到stage暂存区，可以使用git checkout --  filename进行撤销修改。
错误修改如果已经add到stage暂存区，可以使用git reset HEAD filename将该文件重新放回workspace
如果将错误修改已经commit到本地版本库，使用git reset version_number 进行历史穿越。
因此git reset不光可以用于stage的恢复，也可以用于整个版本库的回溯。但如果将修改提交到了远程版本库，就可能会被其他人看到。
/*————————如果文件误删怎么办——————————*/
在工作区删除了某个文件，但是还没有add到stge，想要恢复，使用git checkout -- filename。其实git checkout的作用，就是把本地版本库的最近一次的备份（检查点）同步到workspace，道理和之前文件内容作了修改但还没有add是一样的。
/*———————git远程仓库：Github——————————*/
为了在多台机器上协同工作，可以将本地的仓库托管到远程服务器，而后在另一个本地上从远程服务器同步该仓库，从而使得多个本地上存在仓库的备份。同时如果是多人协作场景，则每个人在远程仓库上的提交都可以共享。
首先需要绑定Github上面的仓库和本地仓库之间的关系：git remote add origin git@github.com:user_name/repo_name
上面命令中，origin表示远程库的名字，是默认习惯命名。
接着使用git push -u origin master将本地仓库推送到关联的远程库上。其中-u表示第一次推送时，git会将本地master分支推动到远程master分支，并且将master分支也关联起来，在之后的推送中就可以简化命令。以后只要git push origin master即可完成本地到远程的推送。
最后，如果需要解除本地库和远程库的绑定，使用git remote rm repo_name。注意远程库并没有被删除，只是断开了绑定，真正删除需要上github手动操作。
/*——————————分支管理————————————*/
对分支的理解，是时间线上的一些结点。每次commit就会在当前分支上产生一个新的结点，之前所学的reset就是在这些结点上进行转移。
master是一个主线分支，在仓库创建的时候自带该分支，所以在仓库上的改动首先默认是在master分支上进行的，名为master的结点会随着改动。在仓库中有个HEAD指针，永远指向当前的活动结点。
创建新分支后，该分支对应前进结点是当前结点；将活动转移到该分支之后，HEAD指针就指向当前分支的前进结点。
git branch <branch_name>创建分支
git switch <branch_name>切换到指定分支
以上两步骤也可合并为一步：git checkout -b <branch_name>
查看当前仓库的分支情况：git branch
切换branch:git switch <branch_name>，其实就是将HEAD指针指向不同branch的前进结点。
合并其他branch到当前branch：git merge <branch_name>
删除其他branch：git branch -d <branch_name>
/*————————冲突解决————————*/
所谓冲突是指两个分支在公共结点之后有岔路存在，这时调用git merge <branch_name>就会合并失败，因为不同岔路会有不同的文件存在。git merge会提示有哪些文件冲突，之后直接打开文件可以看到不同分支下的情况，直接在文件里消除冲突，之后即可合并。
/*—————————管理分支策略—————————*/
如果当前分支与其他分支没有冲突，直接合并默认采用的是Fast-Forward，即快速前进。这样在仓库中会丢弃“被合并分支”的commit结点，这些结点都已转移到“当前分支”上。
禁用Fastorward的方法是在合并是加上参数--no-merge即可。
之后可以使用git log --graph --pretty=oneline --abbrev-commit查看线索图
在实际生产工作中，master分支是用于上线的稳定版本；dev一般是不稳定版本；协作用户可以每人都有一个分支，之后合并到dev上。
/*———————————使用分支修复bug———————————*/
如果在master分支上发现bug，则应该优先考虑创建一个issue分支，修复之后合并到master处，之后删除该issue分支即可。
主要解决bug的思想就是这样。但在实际工作中可能会出现这种情况：如果要切换到有bug的master分支要提交，当前工作区还是unclear的，并且无法提交（可能影响他人工作）。这时可以先将工作区储藏起来，这样工作区当前就是clean。
git stash用于储藏工作区，他像一个stack一样把工作区状态压进。
结束bug修复后，恢复工作区的命令。首先git stash list罗列现有工作区。之后可以指定某个工作区进行恢复。
git stash apply stash@{<index>}即可
git stash drop stash@{<index>}删除工作区
/*————————————新特性/功能开发——————————————*/
通常称为之feature分支。依靠以上所说可以新创建、开发、合并、删除。
但是如果feature还没有被合并到master中时，需要抛弃直接删除。使用git branch -d feature不能成功，需要将参数-d换成-D才可
/*——————————多人协作模式————————————————*/
git clone可以将别人的远程仓库直接下载到自己本地，并且建立了同样的本地仓库。该仓库中的分支只和远程hub上的一样，不会获得拥有其他人本地的分支。
在本地创建新的分支后，如果想要推送到远程，直接git push即可。
如果想要同步远程仓库，首先要将本地同名的分支与远程仓库绑定：
git branch --set-upstream-to=origin/<branch_name> branch_name
git pull 将远程仓库所有分支拉到本地。
