# Git 命令

用到的git 命令总结一下：

   

| config                      |                |      |
| --------------------------- | -------------- | ---- |
| system, global, local       |                |      |
| --global core.editor vim    | default editor |      |
| --global merge.tool vimdiff | merge tool     |      |
|                             |                |      |




| resotre \<file\>   | Restore working tree files                                   |                                                              |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| -S, --staged       | restore staged from HEAD                                     |                                                              |
| -s, --source       | restore the working tree files with the content from commit or the given tree | git restore --source master~2 Makefile :: take a file out of another commit |
| -W, --working tree | restore working tree from staged                             | default                                                      |
|                    |                                                              |                                                              |



| reset [\<mode\>] [\<commit\>] |                                           |                |
| ----------------------------- | ----------------------------------------- | -------------- |
| --solt                        | git reset HEAD                            |                |
| --mixed                       | reset HEAD and index                      | (default mode) |
| --hard                        | reset HEAD, index and working tree        |                |
| --merge                       | reset HEAD, index and working tree(merge) |                |
| HEAD \<file\>                 | reset staged file                         |                |
|                               |                                           |                |
|                               |                                           |                |



| revert \<commit\> | Commits to revert by create a new commit. | require working tree to be clean |
| ----------------- | ----------------------------------------- | -------------------------------- |
|                   |                                           |                                  |



| init               | create an empty git repository or reinitialize an existing one |      |
| ------------------ | ------------------------------------------------------------ | ---- |
| --bare             |                                                              |      |
| -b \<branch-name\> | use the specified name for the initial branch in the newly created repository. |      |
|                    |                                                              |      |



| clone              |                          |      |
| ------------------ | ------------------------ | ---- |
| \<repo\> [\<dir\>] | clone repo to dir        |      |
| --bare             | create a bare repository |      |
|                    |                          |      |



| add          |                                                  |      |
| ------------ | ------------------------------------------------ | ---- |
| *, .         | stage all                                        |      |
| -u, --update | update tracked files                             |      |
| -A, --all    | add changes from all tracked and untracked files |      |
|              |                                                  |      |



| status | show the working tree status |      |
| ------ | ---------------------------- | ---- |
|        |                              |      |



| diff [\<file\>]         | compare working and index               |      |
| ----------------------- | --------------------------------------- | ---- |
| \<branch1\> \<branch2\> | compare between two branches            |      |
| --staged                | comparte between staged and last commit |      |
|                         |                                         |      |



| commit              |                                    |      |
| ------------------- | ---------------------------------- | ---- |
| -m "commit message" |                                    |      |
| -a                  | git add *tracked* files and commit |      |
|                     |                                    |      |



| rm       | Remove files from the working tree and from the index |      |
| -------- | ----------------------------------------------------- | ---- |
| --cached | only remove from the index                            |      |
|          |                                                       |      |



| branch                            | list all local branches                      |      |
| --------------------------------- | -------------------------------------------- | ---- |
| \<branch-name\> [\<start-point\>] | create a new branch at start point           |      |
| -a                                | list both remote-tracking and local branches |      |
| -d                                | delete fully merged branch                   |      |
| -m                                | move/rename a branch and its reflog          |      |
| -r                                | show remote branches                         |      |
| -l                                | list branches                                |      |
|                                   |                                              |      |



| checkout      | compare current branch and remote branch | checkout will be replace by *switch* and *resotre*           |
| ------------- | ---------------------------------------- | ------------------------------------------------------------ |
| \<branch\>    | switch to \<branch\>                     | Local modifications to the files in the working tree are kept, so that they can be committed to the \<branch\> |
| -b \<branch\> | create and checkout a new branch         |                                                              |
| -d \<commit\> | detach HEAD at named commit              |                                                              |
|               |                                          |                                                              |




| switch \<branch\>              | Switch to a specified branch.                                | Switching branches does not require a clean index and working tree |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| --discard-changes, --force, -f | Proceed even if the *index* or the *working tree* differs from HEAD. |                                                              |
| --merge, -m                    | merge if local modifications in *working tree*               |                                                              |
|                                |                                                              |                                                              |



| log       | show commit logs         |      |
| --------- | ------------------------ | ---- |
| -n        | show latest *n* commits  |      |
| --stat    | show statics             |      |
| --oneline | only one line per commit |      |
|           |                          |      |



| stash | Stash the changes in a dirty working directory away          |                             |
| ----- | ------------------------------------------------------------ | --------------------------- |
| list  | List the stash entries that you currently have.              |                             |
| show  | Show the changes recorded in the stash entry                 |                             |
| pop   | Remove a single stashed state from the stash list and apply it on top of the current working tree state |                             |
| apply | Like *pop*, but do not remove the state from the stash list. |                             |
| drop  | drop a stash                                                 | git stash drop  'stash@{0}' |
|       |                                                              |                             |



| tag                       | list all tags                                  |      |
| ------------------------- | ---------------------------------------------- | ---- |
| -a \<tagname\>            | Make an unsigned, annotated tag object         |      |
| -m \<msg\>                | Use the given tag message                      |      |
| -d \<tagname\>            | delete a existing tag                          |      |
| -l [\<pattern\>]          | list only the tags [that match the pattern(s)] |      |
|                           |                                                |      |
| git push origin [tagname] |                                                |      |
| git push origin --tags    |                                                |      |
|                           |                                                |      |



| fetch                 | Download objects and refs from another repository | \<origin\>  default |
| --------------------- | ------------------------------------------------- | ------------------- |
| \<origin\> \<master\> |                                                   |                     |
|                       |                                                   |                     |



| pull [\<options\>] [\<repository\> [\<refspec\>…]] | Fetch from and integrate with another repository or a local branch | git fetch + git merge |
| -------------------------------------------------- | ------------------------------------------------------------ | --------------------- |
| --rebase                                           | git fetch + git rebase                                       |                       |
|                                                    |                                                              |                       |



| push  \<remotehostname> \<localbranch>:\<remotebranch> | Update remote refs along with associated objects             |      |
| ------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| --all                                                  | push all branches                                            |      |
| -u, --set-upstream                                     | For every branch that is up to date or successfully pushed, add upstream (tracking) reference |      |
| git push origin [branch name]                          |                                                              |      |
|                                                        |                                                              |      |



| remote                                   | list tracked repositories                                 |      |
| ---------------------------------------- | --------------------------------------------------------- | ---- |
| -v, -verbose                             | Be a little more verbose and show remote url after name.  |      |
| add \<shortnam\> \<url\>                 | Add a remote named \<name\> for the repository at \<url\> |      |
| rm \<name\>                              | Remove the remote named \<name\>                          |      |
| rename \<old\> \<new\>                   | Rename the remote named \<old\> to \<new\>.               |      |
| set-url \<name\> \<newurl\> [\<oldurl\>] | Changes URLs for the remote.                              |      |
|                                          |                                                           |      |



| rebase | Reapply commits on top of another base tip |      |
| ------ | ------------------------------------------ | ---- |
|        |                                            |      |



| clean | Remove untracked files from the working tree |      |
| ----- | -------------------------------------------- | ---- |
|       |                                              |      |

