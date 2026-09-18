import { chromium } from 'playwright'
import fs from 'node:fs'
fs.mkdirSync('screenshots',{recursive:true})
const browser=await chromium.launch({headless:true})
const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1})

async function shot(path,name){
  await page.goto('http://127.0.0.1:5173'+path,{waitUntil:'networkidle'})
  await page.screenshot({path:'screenshots/'+name+'.png',fullPage:true})
}
async function loginAs(username){
  await page.goto('http://127.0.0.1:5173/login',{waitUntil:'networkidle'})
  await page.locator('input').nth(0).fill(username)
  await page.locator('input').nth(1).fill('123456')
  await page.getByRole('button',{name:'登录',exact:true}).click()
  await page.waitForTimeout(900)
}
async function logoutStorage(){await page.evaluate(()=>{localStorage.clear()})}

await logoutStorage(); await shot('/','01-home')
await shot('/works','02-works')
await shot('/works/1','03-work-detail')
await shot('/register','04-register')

await loginAs('user')
await shot('/custom','05-custom-user')
await shot('/orders','06-orders-user')

await logoutStorage(); await loginAs('creator')
await shot('/creator','07-creator')

await logoutStorage(); await loginAs('admin')
await shot('/admin','08-admin')

await browser.close()
