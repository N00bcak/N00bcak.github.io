import{d as i,c as e,o as t,j as l,k as o,G as p,H as d,e as c,t as a}from"./chunks/framework.DsG8PLbD.js";const r=JSON.parse("[]"),m={key:0,class:"post-empty"},u={key:1,class:"post-list"},g={key:0,class:"post-date"},v=JSON.parse('{"title":"Blog","description":"","frontmatter":{"title":"Blog"},"headers":[],"relativePath":"blog/index.md","filePath":"blog/index.md"}'),_={name:"blog/index.md"},x=i({..._,setup(h){return(y,s)=>(t(),e("div",null,[s[1]||(s[1]=l("p",null,"Welcome to my blog! Here I share my thoughts on various topics including technology, programming, and personal experiences.",-1)),o(r).length===0?(t(),e("div",m,[...s[0]||(s[0]=[l("p",null,"No blogposts live here yet. Stay tuned!",-1)])])):(t(),e("div",u,[(t(!0),e(p,null,d(o(r),n=>(t(),e("article",{key:n.url,class:"post-item"},[n.displayDate?(t(),e("p",g,a(n.displayDate),1)):c("",!0),l("pre",null,[l("code",null,`<h2 class="post-title">
  <a :href="post.url">
    `+a(n.title)+`
  </a>
</h2>

<div
  v-if="post.excerptHtml"
  class="post-excerpt"
  v-html="post.excerptHtml"
/>
`,1)])]))),128))]))]))}});export{v as __pageData,x as default};
