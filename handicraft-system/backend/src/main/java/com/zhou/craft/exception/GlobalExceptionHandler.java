package com.zhou.craft.exception;
import org.springframework.http.*;import org.springframework.web.bind.annotation.*;import java.util.*;
@RestControllerAdvice public class GlobalExceptionHandler{
@ExceptionHandler(IllegalStateException.class)public ResponseEntity<?> state(IllegalStateException e){return ResponseEntity.badRequest().body(Map.of("success",false,"message",e.getMessage()));}
@ExceptionHandler(NoSuchElementException.class)public ResponseEntity<?> missing(NoSuchElementException e){return ResponseEntity.status(404).body(Map.of("success",false,"message","数据不存在"));}
}