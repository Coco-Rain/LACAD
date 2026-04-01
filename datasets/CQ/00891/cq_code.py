import cadquery as cq

result = (
cq.Workplane("XY")
.box(20,40,5)
.faces(">Z").workplane(offset=-5)
.center(15,0).circle(18).extrude(5)
.faces(">Z")
.workplane(centerOption="ProjectedOrigin")
.hole(16*2)
)
cq.exporters.export(result, 'GT.stl')