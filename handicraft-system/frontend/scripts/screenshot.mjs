import { chromium } from 'playwright'
import fs from 'node:fs'

fs.mkdirSync('screenshots',{recursive:true})
const browser=await chromium.launch({headless:true})
const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1})
const base='http://127.0.0.1:5173'

async function shot(path,name){
  await page.goto(base+path,{waitUntil:'networkidle'})
  await page.screenshot({path:'screenshots/'+name+'.png',fullPage:true})
}
async function clearSession(){
  await page.goto(base+'/',{waitUntil:'domcontentloaded'})
  await page.evaluate(()=>localStorage.clear())
}
async function loginAs(username){
  await page.goto(base+'/login',{waitUntil:'networkidle'})
  await page.locator('input').nth(0).fill(username)
  await page.locator('input').nth(1).fill('123456')
  await page.getByRole('button',{name:'登录',exact:true}).click()
  await page.waitForTimeout(1000)
}

await clearSession()
await shot('/','01-home')
await shot('/works','02-works')
await shot('/works/1','03-work-detail')
await shot('/register','04-register')

await loginAs('user')
await shot('/custom','05-custom-user')
await shot('/orders','06-orders-user')

await clearSession()
await loginAs('creator')
await shot('/creator','07-creator')

await clearSession()
await loginAs('admin')
await shot('/admin','08-admin')

await browser.close()
