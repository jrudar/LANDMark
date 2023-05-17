# Welcome to the `LANDMark` contributing guide <!-- omit in toc -->

Thank you for taking the time to contribute to our project!

We ask that all contributors remain respectful and curtious to one another.

## Resources

An overview of `LANDMark` is provided in our [README](https://github.com/jrudar/LANDMark/README.md). 

Here are some resources, provided by GitHub, to help you get started with open source contributions:

- [Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)
- [Set up Git](https://docs.github.com/en/get-started/quickstart/set-up-git)
- [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)

### Issues

#### Create a new issue

If you spot a problem, [search if an issue already exists](https://docs.github.com/en/github/searching-for-information-on-github/searching-on-github/searching-issues-and-pull-requests#search-by-the-title-body-or-comments). If an doesn't exist, you can open a new [issue](https://github.com/jrudar/LANDMark/issues/new). We will try to address your issue in a timely manner.

#### Solve an issue

To solve an issue, you can explore current [existing issues](https://github.com/jrudar/LANDMark/issues/) to find a problem that suits your skill-level. If you find an issue to work on, you are welcome to open a pull-request with a fix. If you are discussing or collaborating on the issue with others, we ask that you remain respectful throughout the collaboration and/or discussion.

### Make Changes

#### Make changes locally and commit your update

Create a working branch by first forking the repository. Once you are happy with the changes made, they are ready for uploading.

### Pull Request

When you're finished with the changes, create a pull request, also known as a PR.
- Provide information about the changes to help us understand your changes as well as the purpose the pull request.
- Don't forget to [link PR to issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) if you are solving one.
- Enable the checkbox to [allow maintainer edits](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork) so the branch can be updated for a merge.
- Once you submit your PR, we may ask questions or request additional information.
- We may suggest additional changes to be made before a PR can be merged.
- As you update your PR and apply changes, mark each conversation as [resolved](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations).
- If you run into any merge issues, checkout this [git tutorial](https://github.com/skills/resolve-merge-conflicts) to help you resolve merge conflicts and other issues.

### Your PR is merged!

If your PR is merged, Congratulations! We thank you for the contribution.

## Working in Windows

This site can be developed on Windows, however a few potential gotchas need to be kept in mind:

1. Regular Expressions: Windows uses `\r\n` for line endings, while Unix-based systems use `\n`. Therefore, when working on Regular Expressions, use `\r?\n` instead of `\n` in order to support both environments. The Node.js [`os.EOL`](https://nodejs.org/api/os.html#os_os_eol) property can be used to get an OS-specific end-of-line marker.
2. Paths: Windows systems use `\` for the path separator, which would be returned by `path.join` and others. You could use `path.posix`, `path.posix.join` etc and the [slash](https://ghub.io/slash) module, if you need forward slashes - like for constructing URLs - or ensure your code works with either.
3. Bash: Not every Windows developer has a terminal that fully supports Bash, so it's generally preferred to write [scripts](/script) in JavaScript instead of Bash.
4. Filename too long error: There is a 260 character limit for a filename when Git is compiled with `msys`. While the suggestions below are not guaranteed to work and could cause other issues, a few workarounds include:
    - Update Git configuration: `git config --system core.longpaths true`
    - Consider using a different Git client on Windows

