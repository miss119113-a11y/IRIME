import axios from 'axios'
export const api=axios.create({baseURL:'/api',timeout:10000})
api.interceptors.request.use(c=>{const t=localStorage.getItem('craft-token');if(t)c.headers.Authorization='Bearer '+t;return c})
export const login=d=>api.post('/auth/login',d);export const register=d=>api.post('/auth/register',d);export const getMe=()=>api.get('/auth/me')
export const getWorks=(params={})=>api.get('/works',{params});export const getWork=id=>api.get('/works/'+id);export const getFeatured=()=>api.get('/works/featured')
export const getFavoriteStatus=id=>api.get('/favorites/work/'+id);export const toggleFavorite=id=>api.post('/favorites/work/'+id);export const getFavorites=()=>api.get('/favorites')
export const getReviews=id=>api.get('/reviews/work/'+id);export const saveReview=(id,d)=>api.post('/reviews/work/'+id,d)
export const getRequests=(params={})=>api.get('/requests',{params});export const createRequest=d=>api.post('/requests',d);export const getQuotes=requestId=>api.get('/quotes',{params:{requestId}});export const createQuote=d=>api.post('/quotes',d);export const createOrderFromQuote=id=>api.post('/orders/from-quote/'+id);export const getOrders=(params={})=>api.get('/orders',{params});export const updateProgress=(id,d)=>api.put('/orders/'+id+'/progress',d);export const completeOrder=id=>api.put('/orders/'+id+'/complete');export const getDashboard=()=>api.get('/dashboard')
