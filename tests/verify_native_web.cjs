// Real manager API + real native URDF/meshes; no API fixtures.
const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs=require('fs');
(async()=>{
 const url=process.env.Q1_NATIVE_WEB_URL || 'http://127.0.0.1:8875';
 const out=process.env.Q1_NATIVE_WEB_OUTPUT || '/tmp/q1_native_web';fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({executablePath:process.env.CHROME_BIN,headless:true,
   args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const errors=[],urdfs=[];page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>{if(r.url().includes('.urdf'))urdfs.push(r.url());});
 await page.goto(url+'/web/',{waitUntil:'domcontentloaded',timeout:120000});
 await page.waitForFunction(()=>document.querySelectorAll('#joints input[type=range]').length===27,{},{timeout:120000});
 await page.waitForSelector('#isaac-status.online',{timeout:60000});
 const state=await (await page.request.get(url+'/api/state')).json();
 const expect=(v,m)=>{if(!v)throw Error(m);};
 expect(state.environment==='ManagerBasedRLEnv','Wrong environment workflow');
 expect(state.contract==='q1-native-manager-v2-23x90','Wrong policy contract');
 expect(state.action_dim===23 && state.observation_dim===90,'Wrong dimensions');
 expect(Object.keys(state.joints).length===27,'Wrong articulation joints');
 expect(!Object.keys(state.joints).some(n=>n.includes('neck')),'Legacy neck slot leaked');
 expect(urdfs.length>0 && urdfs.every(u=>u.includes('/urdf/q1_inertia/')),'Legacy URDF loaded');
 expect(await page.locator('.joint.mimic').count()===4,'Four followers must be read-only');
 await page.locator('#pose-stand').click();
 await page.keyboard.press('KeyW');await page.waitForTimeout(500);await page.keyboard.press('Space');
 const final=await (await page.request.get(url+'/api/state')).json();
 expect(final.environment==='ManagerBasedRLEnv','Web command changed workflow');
 expect(errors.length===0,'Browser errors: '+errors.join(';'));
 await page.screenshot({path:out+'/native_desktop.png'});
 await page.setViewportSize({width:390,height:844});await page.screenshot({path:out+'/native_mobile.png'});
 const result={manager_based:true,native_urdf:true,action_dim:23,observation_dim:90,joints:27,mimic:4,
   urdf_urls:urdfs,mass_kg:state.mass_kg,models:state.models,page_errors:errors};
 fs.writeFileSync(out+'/result.json',JSON.stringify(result,null,2));console.log('NATIVE_WEB_PASS',JSON.stringify(result));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
