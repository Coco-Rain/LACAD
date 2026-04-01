import cadquery as cq

result = (
cq.Workplane("XY").sphere(10)
.workplane(centerOption="ProjectedOrigin").circle(5).cutThruAll()
.workplane(centerOption="ProjectedOrigin").transformed(rotate=(90, 0, 0))
.circle(5).cutThruAll()
.workplane(centerOption="ProjectedOrigin").transformed(rotate=(0, 90, 0))
.circle(5).cutThruAll()
)
cq.exporters.export(result, 'GT.stl')