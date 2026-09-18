<script setup>
import {reactive} from 'vue'
import {login} from '../api'
import {ElMessage} from 'element-plus'
import {useRoute,useRouter} from 'vue-router'
const router=useRouter(),route=useRoute(),form=reactive({username:'user',password:'123456'})
async function submit(){
  try{
    const {data}=await login(form)
    localStorage.setItem('craft-user',JSON.stringify(data))
    localStorage.setItem('craft-token',data.token)
    ElMessage.success('登录成功')
    router.push(route.query.redirect || (data.role==='ADMIN'?'/admin':data.role==='CREATOR'?'/creator':'/'))
  }catch(e){ElMessage.error(e.response?.data?.message||'登录失败')}
}
function fill(u){form.username=u;form.password='123456'}
</script>
<template><div class="login-page"><div class="login-card"><div class="eyebrow">WELCOME BACK</div><h2>登录拾作</h2><p style="font-size:13px;color:#7a7169">进入你的手作世界，继续追踪定制与创作。</p><div class="hint">演示账号：user / creator / admin，密码均为 123456</div><el-form label-position="top"><el-form-item label="用户名"><el-input v-model="form.username" size="large"/></el-form-item><el-form-item label="密码"><el-input v-model="form.password" size="large" type="password" show-password/></el-form-item></el-form><button class="primary-btn" style="width:100%;font-size:14px" @click="submit">登录</button><div style="text-align:center;margin-top:14px;font-size:13px">还没有账号？ <RouterLink to="/register">立即注册</RouterLink></div><div style="display:flex;gap:8px;margin-top:12px"><el-button size="small" @click="fill('user')">普通用户</el-button><el-button size="small" @click="fill('creator')">创作者</el-button><el-button size="small" @click="fill('admin')">管理员</el-button></div></div></div></template>
