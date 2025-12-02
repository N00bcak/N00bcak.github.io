import{d as i,c as t,o as s,j as l,a as d,k as o,G as p,H as c,e as m,t as n}from"./chunks/framework.BXejxr2x.js";const r=JSON.parse("[]"),u={key:0,class:"post-empty"},g={key:1,class:"post-list"},h={key:0,class:"post-date"},b=JSON.parse('{"title":"Blog","description":"","frontmatter":{"title":"Blog"},"headers":[],"relativePath":"blog/index.md","filePath":"blog/index.md"}'),y={name:"blog/index.md"},v=i({...y,setup(_){return(f,e)=>(s(),t("div",null,[e[1]||(e[1]=l("h1",{id:"blog",tabindex:"-1"},[d("Blog "),l("a",{class:"header-anchor",href:"#blog","aria-label":"Permalink to “Blog”"},"​")],-1)),e[2]||(e[2]=l("p",null,"Welcome to my blog! Here I share my thoughts on various topics including technology, programming, and personal experiences.",-1)),o(r).length===0?(s(),t("div",u,[...e[0]||(e[0]=[l("p",null,"No blogposts live here yet. Stay tuned!",-1)])])):(s(),t("div",g,[(s(!0),t(p,null,c(o(r),a=>(s(),t("article",{key:a.url,class:"post-item"},[a.displayDate?(s(),t("p",h,n(a.displayDate),1)):m("",!0),l("pre",null,[l("code",null,`<h2 class="post-title">
  <a :href="post.url">
    `+n(a.title)+`
  </a>
</h2>

<div
  v-if="post.excerptHtml"
  class="post-excerpt"
  v-html="post.excerptHtml"
/>
`,1)])]))),128))]))]))}});export{b as __pageData,v as default};
