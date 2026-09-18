package com.zhou.craft.repo;
import com.zhou.craft.model.CustomRequest;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
public interface CustomRequestRepository extends JpaRepository<CustomRequest,Long>{
    List<CustomRequest> findAllByOrderByCreatedAtDesc();
    List<CustomRequest> findByUserNameOrderByCreatedAtDesc(String userName);
}
